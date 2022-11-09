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
from utils import assign_embedding_net, get_prior_from_params, sumstat_funcs_dict, verify_sumstat, train_model


def append_simulations_from_dir(batch_path, inference, sumstat):
    x = torch.Tensor(torch.load(os.path.join(batch_path, 'x.pt')))
    x = sumstat_funcs_dict[sumstat](x)
    theta = torch.Tensor(torch.load(os.path.join(batch_path, 'theta.pt')))
    inference = inference.append_simulations(theta, x)
    return inference

def main(training_set_path, summary_statistic, output_dir):

    verify_sumstat(summary_statistic)
    embed, embedding_net = assign_embedding_net(summary_statistic)

    os.makedirs(output_dir, exist_ok=False)

    with open(os.path.join(training_set_path,'params.txt'), 'r') as infile:
        params = json.load(infile)

    params['training_set_path'] = training_set_path
    if embed:
        params['nn'] = str([x for x in embedding_net.modules() if not isinstance(x, nn.Sequential)])

    with open(os.path.join(output_dir,'params.txt'), 'w') as outfile:
        json.dump(params, outfile)
        
    prior = get_prior_from_params(params, readable=False)


    for batch_name in os.listdir(training_set_path):
        if batch_name == 'params.txt':
            continue
        batch_num = batch_name.split('_')[1]
        output_path = os.path.join(output_dir,'model_'+batch_num+'.dill')
        batch_path = os.path.join(training_set_path, batch_name)
        density_estimator = 'maf'
        if embed:
            density_estimator = sbiutils.posterior_nn(model='maf', embedding_net=embedding_net)
        inference = SNPE(prior=prior, density_estimator=density_estimator)
        inference = append_simulations_from_dir(batch_path, inference, sumstat=summary_statistic)
        train_model(inference, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Path to output directory of simulations")
    parser.add_argument("-t", "--training_set_path", required=True)
    parser.add_argument("-s", "--summary_statistic", required=True)                        
    args = vars(parser.parse_args())
    main(output_dir=args['output_dir'], training_set_path=args['training_set_path'], 
         summary_statistic=args['summary_statistic'])



