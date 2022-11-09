import argparse
import json
import numpy as np
import torch
import os
from params import syn_prob_coding_only as syn_prob
from evolutionary_model import wrangle_data, get_mutations, simulate_next_passage, normalize_freqs_dict
import sys
sys.path.append('..')
from utils import manual_sumstat_to_dataframe, grab_man_sumstat, get_total_sumstat


def simulate_sequence_sampling(passages, sample_sizes, seq_error_rate, syn_prob, p_ada_syn, p_ada_non_syn, default_sample_size):
    sequenced_passages = dict()
    fitness_effects = np.ones(5)
    for i in passages.keys():
        if i==3:
            sample_size = sample_sizes[0]
        elif i==7:
            sample_size = sample_sizes[1]
        elif i==10:
            sample_size = sample_sizes[2]
        else:
            sample_size = default_sample_size
        seq_errors = get_mutations(sample_size*seq_error_rate, syn_prob, p_ada_syn, p_ada_non_syn, 
                            1/(100*sample_size))
        seq_errors = {(0,0)+mut :freq for mut, freq in seq_errors.items()}
        sequenced_passages[i] = normalize_freqs_dict(passages[i], sample_size)
        if len(sequenced_passages[i])!=0:
            sequenced_passages[i] = simulate_next_passage(fitness_effects, sequenced_passages[i], seq_errors, sample_size, 0)
    return sequenced_passages

def main(input_simulations_path, output_simulations_path, seq_error_rate=0.00005):
    
    DEFAULT_SAMPLE_SIZE = 2000
    
    batches = [x for x in os.listdir(input_simulations_path) if 'batch' in x]
    for batch in batches:
        batch_path = os.path.join(input_simulations_path, batch)
        xs = np.array(grab_man_sumstat(torch.load(os.path.join(batch_path, 'x.pt'))))
        thetas = torch.load(os.path.join(batch_path,'theta.pt'))
        for replica in ['A','B','C']:
            if replica=='A':
                from params import sample_sizes_A as sample_sizes
            elif replica=='B':
                from params import sample_sizes_B as sample_sizes
            elif replica=='C':
                from params import sample_sizes_C as sample_sizes
            res = list()
            for i, (x, theta) in enumerate(zip(xs, thetas)):
                df = manual_sumstat_to_dataframe(x) 
                df.index = [(0,0) + x for x in df.index]
                p_ada_syn = theta[4]
                p_ada_non_syn = theta[5]
                passages = df[[3,7,10]].to_dict()
                seq_passages = simulate_sequence_sampling(passages, sample_sizes, seq_error_rate, syn_prob, p_ada_syn, 
                p_ada_non_syn, DEFAULT_SAMPLE_SIZE)
                res.append(np.array(get_total_sumstat(wrangle_data(seq_passages))))
            new_xs = torch.Tensor(res)
            new_path = os.path.join(output_simulations_path,replica)
            new_batch_path = os.path.join(new_path, batch)
            os.makedirs(new_batch_path)
            torch.save(new_xs, os.path.join(new_batch_path,'x.pt'))
            torch.save(thetas, os.path.join(new_batch_path,'theta.pt'))
            if batch=='batch_0':
                with open(os.path.join(input_simulations_path,'params.txt'), 'r') as infile:
                    params = json.load(infile)
                params['sequence_errors'] = seq_error_rate
                with open(os.path.join(new_path,'params.txt'), 'w') as outfile:
                    json.dump(params, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_simulations_path", required=True,
                        help="Path to output directory of simulations")
    parser.add_argument("-i", "--input_simulations_path", required=True,
                        help="Path to input directory of simulations")
    parser.add_argument("-e", "--seq_error_rate", required=True, type=float,
                        help='sequencing error rate to simulate')
    args = vars(parser.parse_args())
    main(output_simulations_path=args['output_simulations_path'], input_simulations_path=args['input_simulations_path'], 
         seq_error_rate=args['seq_error_rate'])


