import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from utils import get_prior_from_params, grab_short_sumstat, grab_long_sumstat, grab_man_sumstat, calc_stats, verify_sumstat
import time


def get_sims_from_path(path):
    xs = []
    thetas = []
    for batch in os.listdir(path):
        if 'params' in batch:
            continue
        x = torch.load(f'{path}/{batch}/x.pt')
        theta = torch.load(f'{path}/{batch}/theta.pt')
        thetas.append(theta)
        xs.append(x)
    xs = torch.cat(xs)
    thetas = torch.cat(thetas)
    return xs, thetas

def scaled_rmse(simulated_sumstats, empiric_sumstat):
    # insert real sumstat as last row
    big_arr = np.insert(simulated_sumstats,len(simulated_sumstats),empiric_sumstat, axis=0)
    # normalize together
    normed = (big_arr - big_arr.mean(axis=0)) / big_arr.std(axis=0)
    RMSEs = np.sqrt(np.nanmean((normed[:-1] - normed[-1])**2, axis=1))
    return RMSEs

def test_rej_sampling(x_train, x_test, t_train, t_test, prior, acceptance_rate=0.01):
    data = []
    theta_df = pd.DataFrame(np.array(t_train), columns=prior.keys())
    start = time.time()
    for x, theta in zip(x_test,t_test):
        rmse = scaled_rmse(x_train, x)
        accepted_sims = int(acceptance_rate*len(x_train))
        best_rmses_indices = np.argsort(rmse)[:accepted_sims]
        rmse_post = theta_df.loc[best_rmses_indices]
        rmse_stats = calc_stats(rmse_post, theta, prior)
        rmse_stats['theta'] = '_'.join(str(x) for x in theta)
        data.append(rmse_stats)
    print(time.time()-start)
    grid = pd.concat(data).reset_index(drop=True)
    return grid

def main(sims_path, sumstat, output_path, acceptance_rate=0.01):
    verify_sumstat(sumstat)
    sumstat_funcs_dict = {'short': grab_short_sumstat, 'long': grab_long_sumstat, 'man': grab_man_sumstat}
    train_path = os.path.join(sims_path, 'train')
    x_train, t_train = get_sims_from_path(train_path)
    x_train = sumstat_funcs_dict[sumstat](x_train)
    x_test, t_test = get_sims_from_path(os.path.join(sims_path, 'test'))
    x_test = sumstat_funcs_dict[sumstat](x_test)
    with open(os.path.join(train_path,'params.txt'), 'r') as infile:
        params = json.load(infile)
    prior = get_prior_from_params(params, readable=True)
    rej_test = test_rej_sampling(x_train, x_test, t_train, t_test, prior, acceptance_rate=acceptance_rate)
    rej_test.to_csv(output_path) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--simulations_path", required=True, 
                        help='path to simulations containing subdirs train and test')
    parser.add_argument("-o", "--output_path", required=True,
                        help="Path to output directory of simulations")
    parser.add_argument("-s", "--summary_statistic", required=True,
                        help='summary statistic for the model')
    parser.add_argument("-s", "--acceptance_rate", required=True, type=float,
                        help='REJ-ABC accepetance rate - the posterior will be created\
                              from this fraction of simulations')
                            
    args = vars(parser.parse_args())
    main(output_path=args['output_path'], sims_path=args['sims_path'], 
         sumstat=args['summary_statistic'], acceptance_rate=args['acceptance_rate'])