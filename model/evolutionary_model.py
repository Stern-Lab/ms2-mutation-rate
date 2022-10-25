
# These are all helper functions for the simulator which does little besides importing and using them

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import poisson
import itertools
from scipy.special import factorial
import torch

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

def get_short_sumstat(df, passages=[3,7,10]):
    ret = list()
    for passage in passages:
        ret.append(sum(df['syn_total'] * df[passage]))
        ret.append(sum(df['non_syn_total'] * df[passage]))
    return torch.Tensor(ret)

def get_manual_stats(df, passages=[3,7,10]):
    max_muts = 11 # so 10 maximum muts
    new_index = [(x,y,z,w) for w in range(max_muts) for z in range(max_muts)
                    for y in range(max_muts) for x in range(max_muts) if x+y<max_muts and z<=x and w<=y]
    grouped = df.groupby(['syn_total', 'non_syn_total', 'syn_ben', 'non_syn_ben'])[passages].sum()
    return torch.Tensor(grouped.reindex(new_index).fillna(0).values.flatten())

def get_med_stats(df, passages=[3,7,10]):
    max_muts = 11 # so 10 maximum muts
    new_index = [(x,y) for y in range(max_muts) for x in range(max_muts) if x+y<max_muts]
    grouped = df.groupby(['syn_total', 'non_syn_total'])[passages].sum()
    return torch.Tensor(grouped.reindex(new_index).fillna(0).values.flatten())

def get_total_sumstat(df, passages=[3,7,10]):
    return torch.cat((get_short_sumstat(df, passages), get_med_stats(df, passages), get_manual_stats(df, passages)))

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
