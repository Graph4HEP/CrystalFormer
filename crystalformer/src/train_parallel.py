import jax
import jax.numpy as jnp
from jax import devices, pmap
from functools import partial
import os, sys
import optax
import math

from crystalformer.src.utils import shuffle
import crystalformer.src.checkpoint as checkpoint

import time

def process_bar_train(num, total, dt, loss, loss_w, loss_a, loss_xyz, loss_l, Type=''):
    rate = float(num)/total
    ratenum = int(50*rate)
    estimate = dt/rate*(1-rate)
    r = '\r{} [{}{}]{}/{} - used {:.1f}s / left {:.1f}s / loss total {:.2f} / loss w {:.2f} / loss a {:.2f} / loss xyz {:.2f} / loss l {:.2f} '.format(Type, '*'*ratenum,' '*(50-ratenum), num, total, dt, estimate, loss, loss_w, loss_a, loss_xyz, loss_l)
    sys.stdout.write(r)
    sys.stdout.flush()


def train_mgpu(key, optimizer, opt_state, loss_fn, params, epoch_finished, epochs, batchsize, train_data, valid_data, path, args):
    
    @partial(pmap, axis_name='batch')
    @jax.jit
    def update(params, key, data):
        G, L, X, A, W = data
        G = jnp.squeeze(G)
        L = jnp.squeeze(L)
        A = jnp.squeeze(A)
        W = jnp.squeeze(W)
        value, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, G, L, X, A, W, True)
        #updates, opt_state = optimizer.update(grad, opt_state, params)
        #params = optax.apply_updates(params, updates)
        loss, aux = value
        loss = jax.lax.pmean(loss, axis_name='batch')
        aux = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), aux)
        value = loss, aux
        grads = jax.lax.psum(grads, "batch")
        return grads, value

    @jax.jit
    def backward(sub_grads, opt_state, params):
        updates, opt_state = optimizer.update(sub_grads, opt_state, params)
        params = optax.apply_updates(params, updates)        
        return params, opt_state

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch t_loss v_loss t_loss_w v_loss_w t_loss_a v_loss_a t_loss_xyz v_loss_xyz t_loss_l v_loss_l\n")
    
    st_ep = time.time()
    devices = jax.local_devices() 
    for epoch in range(epoch_finished+1, epochs):
        key, subkey = jax.random.split(key)
        train_data = shuffle(subkey, train_data)

        train_G, train_L, train_X, train_A, train_W = train_data 

        train_loss = 0.0 
        train_aux = 0.0, 0.0, 0.0, 0.0
        num_samples = len(train_L)
        num_batches = round(num_samples / batchsize)-1
        st = time.time()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)            

            data_G = jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), train_G[start_idx:end_idx].reshape(batchsize,-1))
            data_L = jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), train_L[start_idx:end_idx].reshape(batchsize,-1))
            data_X = train_X[start_idx:end_idx].reshape(jax.local_device_count(),round(batchsize/jax.local_device_count()),args.n_max,3)
            data_A = jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), train_A[start_idx:end_idx].reshape(batchsize,-1))
            data_W = jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), train_W[start_idx:end_idx].reshape(batchsize,-1))
            data = data_G, data_L, data_X, data_A, data_W        

            replicated_params = jax.device_put_replicated(params, devices)

            keys = jax.random.split(key, jax.local_device_count()+1)            
            subkey = keys[1:]

            grads, (loss, aux) = update(replicated_params, subkey, data)
            loss = loss[0]
            aux = aux[0][0], aux[1][0], aux[2][0], aux[3][0]
            sub_grads = {}
            for k in grads.keys():
                if(isinstance(grads[k], dict)):
                    for j in grads[k].keys():
                        sub_grads[k] = {}
            for k in grads.keys():
                if(isinstance(grads[k], dict)):
                    for j in grads[k].keys():                        
                        sub_grads[k][j] = grads[k][j][0]
                else:
                    sub_grads[k] = grads[k][0]
            params, opt_state = backward(sub_grads, opt_state, params)

            train_loss, train_aux = jax.tree_map(
                        lambda acc, i: acc + i,
                        (train_loss, train_aux), 
                        (loss, aux)
                        )

            if(batch_idx%10==1 or batch_idx==num_batches-1):
                loss_w, loss_a, loss_xyz, loss_l = train_aux
                loss_w = loss_w/(batch_idx+1)
                loss_a = loss_a/(batch_idx+1)
                loss_xyz = loss_xyz/(batch_idx+1)
                loss_l = loss_l/(batch_idx+1)
                process_bar_train(batch_idx, num_batches-1, time.time()-st, train_loss/(batch_idx+1), loss_w, loss_a, loss_xyz, loss_l)

        train_loss, train_aux = jax.tree_map(
                        lambda x: x/num_batches, 
                        (train_loss, train_aux)
                        ) 
        print(f'\n{epoch}: used time {time.time()-st_ep:.1f}s, left_time {(time.time()-st_ep)/(epoch-epoch_finished)*(10000-epoch):.1f}s')
        f.write( ("%6d" + 10*"  %.6f" + "\n") % (epoch,
                                                train_loss,   0,
                                                loss_w, 0,
                                                loss_a, 0,
                                                loss_xyz, 0,
                                                loss_l, 0
                                                ))
        

        if epoch % 100 == 0 or epoch == epochs-1:
            valid_G, valid_L, valid_X, valid_A, valid_W = valid_data 
            valid_loss = 0.0 
            valid_aux = 0.0, 0.0, 0.0, 0.0
            num_samples = len(valid_L)
            num_batches = math.ceil(num_samples / batchsize)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batchsize
                end_idx = min(start_idx + batchsize, num_samples)
                G, L, X, A, W = valid_G[start_idx:end_idx], \
                              valid_L[start_idx:end_idx], \
                              valid_X[start_idx:end_idx], \
                              valid_A[start_idx:end_idx], \
                              valid_W[start_idx:end_idx]

                key, subkey = jax.random.split(key)
                loss, aux = loss_fn(params, subkey, G, L, X, A, W, False)
                valid_loss, valid_aux = jax.tree_map(
                        lambda acc, i: acc + i,
                        (valid_loss, valid_aux), 
                        (loss, aux)
                        )

            valid_loss, valid_aux = jax.tree_map(
                        lambda x: x/num_batches, 
                        (valid_loss, valid_aux)
                        ) 

            train_loss_w, train_loss_a, train_loss_xyz, train_loss_l = train_aux
            valid_loss_w, valid_loss_a, valid_loss_xyz, valid_loss_l = valid_aux

            f.write( ("%6d" + 10*"  %.6f" + "\n") % (epoch, 
                                                    train_loss,   valid_loss,
                                                    train_loss_w, valid_loss_w, 
                                                    train_loss_a, valid_loss_a, 
                                                    train_loss_xyz, valid_loss_xyz, 
                                                    train_loss_l, valid_loss_l
                                                    ))

            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return params, opt_state

