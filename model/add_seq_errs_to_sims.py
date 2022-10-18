import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import poisson
import itertools
from scipy.special import factorial
import torch
from utils import manual_sumstat_to_dataframe, grab_man_sumstat, get_total_sumstat
import shutil
import os
from params import syn_prob_coding_only as syn_prob


def main(input_simulations_path, output_simulations_path, seq_error_rate=0.00005):
    
    if os.path.exists(output_simulations_path):
        raise Exception(f'{output_simulations_path} exists! please choose a new output dir.')
    
    default_sample_size = 2000
    
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
                print(i)
                seq_passages = simulate_sequence_sampling(passages, sample_sizes, seq_error_rate, syn_prob, p_ada_syn, 
                p_ada_non_syn, default_sample_size)
                res.append(np.array(get_total_sumstat(wrangle_data(seq_passages))))
            new_xs = torch.Tensor(res)
            new_path = os.path.join(output_simulations_path,f'replica_{replica}_err_rate_{seq_error_rate}',batch)
            os.makedirs(new_path)
            torch.save(new_xs, os.path.join(batch_path,'/x.pt'))
            torch.save(thetas, os.path.join(batch_path,'/theta.pt'))
            if batch=='batch_0':
                shutil.copy(os.path.join(input_simulations_path,'params.txt'), os.path.join(new_path,'params.txt'))

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

# The functions below are copied(!!!) from evolutionary_model.py
# this is because dill requires functions (and imports) to be nested within the simulator... 

def multinomial_sampling(freqs_dict, sample_size):
    freqs_after_sample = np.random.multinomial(sample_size, list(freqs_dict.values()))/sample_size
    freqs_dict = {key: val for key, val in zip(freqs_dict.keys(), freqs_after_sample) if val>0}
    return freqs_dict

def get_poisson_probs(mutation_rate, min_freq):
    probs = dict()
    for num_of_muts in range(10000):  # just a large number coz while loops are slow and awkward
        prob = poisson.pmf(num_of_muts, mutation_rate)
        if prob <= min_freq:
            break
        probs[num_of_muts] = prob
    return probs

def generate_possible_GCs(mut_num):
    GCs_unformatted = itertools.combinations_with_replacement(['syn', 'non_syn', 'syn_ada', 'non_syn_ada'], 
                                                              mut_num)
    possible_GCs = list()
    for muts_combo in GCs_unformatted:
        possible_GCs.append((muts_combo.count('syn'), muts_combo.count('non_syn'),
                             muts_combo.count('syn_ada'), muts_combo.count('non_syn_ada')))
    return possible_GCs

def calc_mutations_probs(mut_num, poisson_prob, ps, min_freq):
    GCs_array = generate_possible_GCs(mut_num)
    probabilities = np.product(np.power(ps, GCs_array), axis=1) 
    factorials = factorial(mut_num)/np.product(factorial(GCs_array), axis=1)
    multinomial_prob = factorials * probabilities * poisson_prob
    return {GC: prob for GC, prob in zip(GCs_array, multinomial_prob) if prob>min_freq}

def get_mutations(mutation_rate, syn_ratio, p_ada_syn, p_ada_non_syn, min_freq):
    ps = [syn_ratio - p_ada_syn, 
          1 - syn_ratio - p_ada_non_syn, 
          p_ada_syn, 
          p_ada_non_syn]
    mut_poisson_prob = get_poisson_probs(mutation_rate, min_freq)
    mutations = dict()
    for mut_num, poisson_prob in mut_poisson_prob.items():
        mutations.update(calc_mutations_probs(mut_num, poisson_prob, ps, min_freq))
    return mutations

def simulate_p0(p0_syn, p0_non_syn, pop_size):
    p0_sum_of_mutations = p0_syn + p0_non_syn
    p0_syn_ratio = p0_syn / p0_sum_of_mutations
    # we assume no adaptive mutations at p0
    p0_mutations = get_mutations(p0_sum_of_mutations, p0_syn_ratio, 0, 0, 1/(100*pop_size))
    # convert to 6-tuple 
    p0_mutations = {mut+(0,0) :freq for mut, freq in p0_mutations.items()}
    return p0_mutations

def get_epistatsis(muts_by_fitness, epistasis_boost):
    # give penalty value to genotypes with more than one adaptive mutation
    multiple_adaptive_idx = np.argwhere(muts_by_fitness[:,4]>1).reshape(-1)
    fitness_len = muts_by_fitness.shape[0]
    epistasis = [epistasis_boost if x in multiple_adaptive_idx else 1 for x in range(fitness_len)]
    return epistasis

def selection(fitness_effects, muts_by_fitness, freqs, epistasis_boost):
    no_epi_fitness = np.product(np.power(fitness_effects, muts_by_fitness), axis=1).reshape(-1)
    epi_effects = fitness_effects.copy()
    epi_effects[4] =  epi_effects[4] ** epistasis_boost
    epistasis_fitness = np.product(np.power(epi_effects, muts_by_fitness), axis=1).reshape(-1)
    fitness = np.where(muts_by_fitness[:,4]>1, epistasis_fitness, no_epi_fitness)
    avg_fitness = np.sum(freqs*fitness)
    fitness /= avg_fitness
    return fitness

def gather_muts_by_fitness(genotypes):
    primordial = genotypes[:,:2]
    no_adas = genotypes[:,2:4]
    just_adas = np.sum(genotypes[:,4:], axis=1).reshape(-1,1)
    return np.concatenate([primordial, no_adas, just_adas], axis=1)

def mutate_and_select(genotypes, genotypes_freqs, mutations, mutations_freqs, fitness_effects, 
                      tuple_size, epistasis_boost):
    # do that numpy magic:
    new_genotypes = genotypes + mutations
    new_genotypes = new_genotypes.reshape(-1,tuple_size)
    new_freqs = genotypes_freqs * mutations_freqs                          # mutation
    new_freqs = new_freqs.reshape(-1)
    muts_by_fitness = gather_muts_by_fitness(new_genotypes)
    fitness = selection(fitness_effects, muts_by_fitness, new_freqs, epistasis_boost) 
    new_genotypes = list(map(tuple, new_genotypes))
    new_freqs = new_freqs * fitness
    return new_genotypes, new_freqs

def normalize_freqs_dict(freqs_dict, pop_size):
    freqs_dict = {key: val for key, val in freqs_dict.items() if val > 1/(pop_size*1000)}  # to prevent occasional bugs
    freqs_sum = sum(freqs_dict.values())
    freqs_dict = {key: val/freqs_sum for key, val in freqs_dict.items()}
    return freqs_dict
    
def simulate_next_passage(fitness_effects, passage, mutations, pop_size, epistasis_boost):
    # turn dict into arrays:
    tuple_size = len(list(passage.keys())[0])
    genotypes = np.array(list(passage.keys()), dtype=int).reshape(-1,1,tuple_size)
    genotypes_freqs = np.array(list(passage.values()), dtype=float).reshape(-1,1,1)
    new_genotypes, new_freqs = mutate_and_select(genotypes, genotypes_freqs, np.array(list(mutations.keys())), 
                                                 np.array(list(mutations.values())), fitness_effects, 
                                                 tuple_size, epistasis_boost)
    freqs_dict = defaultdict(float)
    for mut, freq in zip(new_genotypes, new_freqs):
        freqs_dict[mut] += freq
    freqs_dict = normalize_freqs_dict(freqs_dict, pop_size)
    freqs_dict = multinomial_sampling(freqs_dict, pop_size)               # drift
    return freqs_dict

def wrangle_data(passage):
    data = pd.DataFrame(passage)
    data['mut_num'] = [sum(x) for x in data.index]
    data = data.reset_index().rename(columns={'level_5': 'non_syn_ben', 'level_1': 'non_syn_pri',
                                              'level_4': 'syn_ben', 'level_0': 'syn_pri',
                                              'level_3': 'non_syn', 'level_2': 'syn'}).fillna(0)
    data['syn_non_ben'] = data['syn'] + data['syn_pri']
    data['non_syn_non_ben'] = data['non_syn'] + data['non_syn_pri']
    data['syn_total'] = data['syn_non_ben'] + data['syn_ben']
    data['non_syn_total'] = data['non_syn_non_ben'] + data['non_syn_ben']
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_simulations_path", required=True,
                        help="Path to output directory of simulations")
    parser.add_argument("-i", "--input_simulations_path", required=True,
                        help="Path to input directory of simulations")
    parser.add_argument("-e", "--seq_error_rate", required=True,
                        help='sequencing error rate to simulate')
    args = vars(parser)
    main(output_simulations_path=args['output_simulations_path'], input_simulations_path=args['input_simulations_path'], 
         seq_error_rate=args['seq_error_rate'])


