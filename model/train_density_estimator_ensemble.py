# this script will train an individual model for each training batch

import argparse
import torch.nn as nn 
from sbi.inference import SNPE
from sbi import utils as sbiutils
import torch
import os
import dill
import json
import sys
sys.path.append('..')
from utils import get_prior_from_params, grab_short_sumstat, grab_long_sumstat, grab_man_sumstat


def append_simulations_from_dir(batch_path, inference, sumstat):
    sumstat_funcs_dict = {'short': grab_short_sumstat, 'long': grab_long_sumstat, 'man': grab_man_sumstat}
    x = torch.Tensor(torch.load(os.path.join(batch_path, 'x.pt')))
    x = sumstat_funcs_dict[sumstat](x)
    theta = torch.Tensor(torch.load(os.path.join(batch_path, 'theta.pt')))
    inference = inference.append_simulations(theta, x)
    return inference

def train_model(inference, output_path, max_epochs=600):
    density_estimator = inference.train(max_num_epochs=max_epochs)
    posterior = inference.build_posterior(density_estimator)
    with open(output_path, "wb") as handle:
        dill.dump(posterior, handle)
    return posterior

def main(training_set_path, output_path, summary_statistic):

    if summary_statistic=='long':
        embed=True
        embedding_net = nn.Sequential(nn.Linear(204, 128), 
                        nn.ReLU(),
                        nn.Linear(128, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16))
    elif summary_statistic=='man':
        embed=True
        embedding_net = nn.Sequential(nn.Linear(3009, 512), 
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128, 16))
    elif summary_statistic=='short':
        embed=False
    else:
        raise Exception(f'summary_statistic should be one of [short, long, man] not {summary_statistic} !')

    inference_dir = os.path.join(output_path, summary_statistic)

    os.makedirs(inference_dir, exist_ok=False)

    with open(os.path.join(training_set_path,'params.txt'), 'r') as infile:
        params = json.load(infile)

    params['training_set_path'] = training_set_path
    if embed:
        params['nn'] = str([x for x in embedding_net.modules() if not isinstance(x, nn.Sequential)])

    with open(os.path.join(inference_dir,'params.txt'), 'w') as outfile:
        json.dump(params, outfile)
        
    prior = get_prior_from_params(params, readable=False)


    for batch_name in os.listdir(training_set_path):
        if batch_name == 'params.txt':
            continue
        batch_num = batch_name.split('_')[1]
        output_path = os.path.join(inference_dir,'model_'+batch_num+'.dill')
        batch_path = os.path.join(training_set_path, batch_name)
        density_estimator = 'maf'
        if embed:
            density_estimator = sbiutils.posterior_nn(model='maf', embedding_net=embedding_net)
        inference = SNPE(prior=prior, density_estimator=density_estimator)
        inference = append_simulations_from_dir(batch_path, inference, sumstat=summary_statistic)
        train_model(inference, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", required=True,
                        help="Path to output directory of simulations")
    parser.add_argument("-t", "--training_set_path", required=True)
    parser.add_argument("-s", "--summary_statistic", required=True)                        
    args = vars(parser.parse_args())
    main(output_path=args['output_path'], training_set_path=args['training_set_path'], 
         summary_statistic=args['summary_statistic'])


