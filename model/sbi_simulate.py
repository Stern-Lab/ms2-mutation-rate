# each training batch will eventually be a separate model thus,
# the number of batches is the final number of models in the ensemble model

import argparse
from sbi.inference import prepare_for_sbi, simulate_for_sbi
from sbi import utils as sbiutils
import torch
import os
import json
import numpy as np
import multiprocessing as mp
from functools import partial
from simulator import simulate
import sys
from params import passages, pop_size, readable_prior
from params import syn_prob_coding_only as syn_prob
sys.setrecursionlimit(100000)  # appearently necessary for dill when doing many batches


def main(output_dir, ensemble_size=7, simulations_per_batch=1000, unitest_run=False):
    # num of simulations = number_of_batches * syms_per_batch 
    os.makedirs(output_dir, exist_ok=True)
    if unitest_run:
        readable_prior['mu'] = (-4,-3)  # makes everything run fast enough for unitesting!
    params = readable_prior.copy()
    params['syn_prob'] = syn_prob
    params['passages'] = passages
    params['pop_size'] = pop_size
    params['simulations_per_batch'] = simulations_per_batch
    with open(f'{output_dir}/params.txt', 'w') as outfile:
        json.dump(params, outfile)

    simulator = partial(simulate, syn_prob=syn_prob, passages=passages,
                        pop_size=pop_size)

    prior = sbiutils.BoxUniform(low=np.array([val[0] for val in readable_prior.values()]), 
                                high=np.array([val[1] for val in readable_prior.values()]))

    dir_paths = [os.path.join(output_dir,f'batch_{x}') for x in range(ensemble_size)]

    simulator, prior = prepare_for_sbi(simulator, prior)

    for tmp_path in dir_paths:
        os.makedirs(tmp_path, exist_ok=False)
        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=simulations_per_batch, num_workers=mp.cpu_count()-1)
        torch.save(x, f'{tmp_path}/x.pt')
        torch.save(theta, f'{tmp_path}/theta.pt')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Path to output directory of simulations")
    parser.add_argument("-e", "--ensemble_size", required=True, type=int,
                        help='number of seperate simulation batches \
                              and size of ensemble density estimator trained on the simulations')
    parser.add_argument("-s", "--simulations_per_batch", required=True, type=int)                        
    args = vars(parser.parse_args())
    main(output_dir=args['output_dir'], ensemble_size=args['ensemble_size'], 
         simulations_per_batch=args['simulations_per_batch'])


