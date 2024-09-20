import sys
sys.path.append('scripts')
from awl2struct import main as awl2s
from compute_metrics import main as comp
from compute_metrics_matbench import main as comp_bench

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_path', default='./model/single_gpu_result/', help='filepath of the output and input file')
    parser.add_argument('--label', default='1', help='output file label')
    parser.add_argument('--num_io_process', type=int, default=20, help='number of process used in multiprocessing io')
    parser.add_argument('--root_path', default='')
    parser.add_argument('--filename', default='')

    parser.add_argument('--train_path', default='./data/mp_20/train.csv', help='')
    parser.add_argument('--test_path', default='./data/mp_20/test.csv', help='')
    parser.add_argument('--gen_path', default='', help='')

    args = parser.parse_args()
    #model output to cifs
    awl2s(args)

    #check cif validity
    args.root_path = args.output_path    
    args.filename = f'output_{args.label}_struct.csv'
    comp(args)

    #check novity and coverage and other index for the gen data
    #args.gen_path = f'{args.output_path}{args.filename}'
    #comp_bench(args)
