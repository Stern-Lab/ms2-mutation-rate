import argparse
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as sbiutils
import torch
import os
import json
import numpy as np
import multiprocessing as mp
from functools import partial
from params import passages, pop_size, genome_length
import sys
from evolutionary_model import simulate
from params import readable_prior
from params import syn_prob_coding_only as syn_prob
sys.setrecursionlimit(100000)  # appearently necessary for dill when doing many batches


def main(output_dir, number_of_batches=7, simulations_per_batch=1000):
    # num of simulations = number_of_batches * syms_per_batch 
    os.makedirs(output_dir, exist_ok=True)
    params = readable_prior.copy()
    params['syn_prob'] = syn_prob
    params['passages'] = passages
    params['pop_size'] = pop_size
    params['genome_length'] = genome_length
    params['simulations_per_batch'] = simulations_per_batch

    with open(f'{output_dir}/params.txt', 'w') as outfile:
        json.dump(params, outfile)

    simulator = partial(simulate, syn_prob=syn_prob, passages=passages,
                        pop_size=pop_size, genome_length=genome_length)

    prior = sbiutils.BoxUniform(low=np.array([val[0] for val in readable_prior.values()]), 
                                high=np.array([val[1] for val in readable_prior.values()]))

    dir_paths = [os.path.join(output_dir,f'batch_{x}') for x in range(number_of_batches)]

    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE(prior)

    for tmp_path in dir_paths:
        os.makedirs(tmp_path, exist_ok=False)
        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=simulations_per_batch, num_workers=mp.cpu_count())
        torch.save(x, f'{tmp_path}/x.pt')
        torch.save(theta, f'{tmp_path}/theta.pt')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Path to output directory of simulations")
    parser.add_argument("-b", "--number_of_batches", required=True)
    parser.add_argument("-s", "--simulations_per_batch", required=True)                        
    args = vars(parser)
    main(output_dir=args['output_dir'], number_of_batches=args['number_of_batches'], 
         simulations_per_batch=args['simulations_per_batch'])


