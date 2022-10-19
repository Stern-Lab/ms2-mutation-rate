import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from utils import get_ensemble_predictions, get_prior_from_params, grab_short_sumstat, grab_long_sumstat, grab_man_sumstat


DEFAULT_SAMPLES_PER_ESTIMATOR = 1000

def main(density_estimators_path, test_set_path, output_dir, samples_per_estimator=DEFAULT_SAMPLES_PER_ESTIMATOR):
    
    sumstat_funcs_dict = {'short': grab_short_sumstat, 'long': grab_long_sumstat, 'man': grab_man_sumstat}
    summary_statistic = os.path.basename(os.path.normpath(density_estimators_path))
    with open(os.path.join(density_estimators_path,'params.txt'), 'r') as infile:
        params = json.load(infile)
    prior = get_prior_from_params(params, readable=True)

    xs = []
    thetas = []
    for batch in os.listdir(test_set_path):
        if 'params' in batch:
            continue
        batch_path = os.path.join(test_set_path,batch)
        x = torch.load(os.path.join(batch_path,'x.pt'))
        theta = torch.load(os.path.join(batch_path,'theta.pt'))
        thetas.append(theta)
        xs.append(x)
    xs = torch.cat(xs)
    xs = sumstat_funcs_dict[summary_statistic](xs)
    thetas = torch.cat(thetas)

    xs = np.around(xs,3)
    thetas = np.around(thetas, 3)

    data = []
    for x, theta in zip(xs,thetas):
        posterior, stats = get_ensemble_predictions(density_estimators_path, x, theta, prior, samples_per_estimator)
        stats['theta'] = '_'.join(str(float(t)) for t in theta)
        data.append(stats)
    grid = pd.concat(data).reset_index(drop=True)
    grid.to_csv(os.path.join(output_dir, 'ensemble_test.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--density_estimators_path", required=True,
                        help="Path to density estimators with a specific summary statistic")
    parser.add_argument("-t", "--test_set_path", required=True)
    parser.add_argument("-o", "--output_dir", required=True) 
    parser.add_argument("-s", "--samples_per_estimator", required=False, type=int,
                        help='number of samples to draw from each estimator', default=DEFAULT_SAMPLES_PER_ESTIMATOR)                        
    args = vars(parser.parse_args())
    main(density_estimators_path=args['density_estimators_path'], test_set_path=args['test_set_path'], 
         output_dir=args['output_dir'], samples_per_estimator=args['samples_per_estimator'])



