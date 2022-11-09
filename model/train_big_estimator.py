import argparse
from sbi.inference import SNPE
from sbi import utils as sbiutils
import torch
import torch.nn as nn 
import os
import dill
import json
import sys
sys.path.append('..')
from utils import get_prior_from_params, sumstat_funcs_dict, verify_sumstat, assign_embedding_net, train_model


def append_sims_from_batches_dir(xs, thetas, batches_dir):
    for batch_name in os.listdir(batches_dir):
        if batch_name == 'params.txt':
            continue
        batch_path = os.path.join(batches_dir, batch_name)
        xs.append(torch.load(os.path.join(batch_path, 'x.pt'))) 
        thetas.append(torch.load(os.path.join(batch_path, 'theta.pt')))
    return xs, thetas


def main(training_path, summary_statistic, output_path):
    verify_sumstat(summary_statistic)
    embed, embedding_net = assign_embedding_net(summary_statistic)
    
    os.makedirs(output_path, exist_ok=False)
    
    with open(os.path.join(training_path,'params.txt'), 'r') as infile:
        params = json.load(infile)
    params['sims_per_model'] = 'all'
    params['batches_dir'] = training_path
    if embed:
        params['nn'] = str([x for x in embedding_net.modules() if not isinstance(x, nn.Sequential)])
    with open(os.path.join(output_path,'params.txt'), 'w') as outfile:
        json.dump(params, outfile)
        
    prior = get_prior_from_params(params, readable=False)

    estimator_path = os.path.join(output_path,'big_estimator.dill')
    xs = []
    thetas = []
    xs, thetas = append_sims_from_batches_dir(xs, thetas, training_path)
    x = torch.cat(xs)
    x = sumstat_funcs_dict[summary_statistic](x)
    theta = torch.cat(thetas)
    density_estimator = 'maf'
    if embed:
        density_estimator = sbiutils.posterior_nn(model='maf', embedding_net=embedding_net)
    inference = SNPE(prior=prior, density_estimator=density_estimator)
    inference = inference.append_simulations(theta, x)
    train_model(inference, estimator_path)


    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--training_path", required=True, 
                            help='path to simulations containing subdirs train and test')
        parser.add_argument("-o", "--output_path", required=True,
                            help="Path to output directory of simulations")
        parser.add_argument("-s", "--summary_statistic", required=True,
                            help='summary statistic for the model')
                                
        args = vars(parser.parse_args())
        main(output_path=args['output_path'], training_path=args['training_path'], 
            sumstat=args['summary_statistic'])