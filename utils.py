import dill
import numpy as np
from sbi import analysis as sbianalysis
from sbi import utils as sbiutils
import pandas as pd
import os
import torch
import torch.nn as nn 
import arviz as az
import numpy as np
import pandas as pd


#TODO: cleanups!

prior_names = ['reg_del_prob', 'reg_ada_prob', 'syn_del_prob', 'syn_ada_prob', 'ada_val', 'del_val', 'mu', 
               'p0_syn', 'p0_non_syn']
prior_names = ['mu', 'w_syn', 'w_ratio', 'w_ada', 'p_ada_syn', 'p_ada_non_syn', 'p0_syn', 'p0_non_syn', 'w_penalty', 'epistasis_boost']
prior_names = ['mu', 'w_syn', 'w_non_syn', 'w_ada', 'p_ada_syn', 'p_ada_non_syn', 'p0_syn', 'p0_non_syn', 'w_penalty', 'epistasis_boost']


class sbi_post():
    
    def __init__(self, density_estimator):
        with open(density_estimator, "rb") as handle:
            inference_from_disk = dill.load(handle)
        self.post = inference_from_disk
        self.posterior = None
        self.labels = prior_names
    
    def generate_post(self, dict_values, num_of_samples=10000):
        self.posterior = self.post.sample((num_of_samples,), np.array(list(dict_values)))
        
    def pairplot(self, data_values=None, num_of_samples=10000):
        if data_values:
            self.generate_post(data_values, num_of_samples)
        _ = sbianalysis.pairplot(self.posterior, figsize=(10,10), labels=self.labels)

    def get_post(self):
        return pd.DataFrame(self.posterior.numpy(), columns=self.labels)
        

def get_tensor_passages(tensor, passages_list=[3,7,10]):
    regs = np.array(passages_list)-1
    syns = regs + 10
    passages = np.concatenate([regs,syns])
    return tensor[:,passages]

def get_mode_and_hdi(data, bins=100, hdi_percent=0.95):
    mode_interval = pd.cut(data, bins).mode().values[0]
    mode = (mode_interval.left + mode_interval.right)/2
    hdi = az.hdi(np.array(data),hdi_percent)
    return mode, hdi


def DKL_prior_post(prior, posterior):
    bin_num = len(posterior)
    Q = np.ones(bin_num)/bin_num                                            # uniform prior
    P = np.histogram(posterior, bin_num, range=prior)[0] / len(posterior)  # normalized posterior
    divergence = np.sum(np.where(P!=0, P*np.log(P/Q), 0))
    return divergence


def calc_stats(post, theta, readable_prior, bins=100):
    stats = []
    cols = [x for x in post.columns if x!='model']
    for i, col in enumerate(cols):
        this_post = post[col]
        this_prior = np.array(readable_prior[col])
        mode, hdi50 = get_mode_and_hdi(this_post, hdi_percent=0.50, bins=bins)
        mode, hdi95 = get_mode_and_hdi(this_post, hdi_percent=0.95, bins=bins)
        if col=='mu':
            this_post = 10 ** this_post
            this_prior = 10 ** this_prior
        dkl = DKL_prior_post(this_prior, this_post)
        tmp_dict = {'param': col, 'mode': mode, 'hdi95_low': hdi95[0], 'DKL': dkl,
                      'hdi95_high': hdi95[1],'hdi50_low': hdi50[0], 'hdi50_high': hdi50[1]}
        if theta is not None:
            tmp_dict['value'] = float(theta[i])
        stats.append(tmp_dict) 
    stats = pd.DataFrame.from_records(stats)
    if theta is not None:
        stats['err'] = stats['mode'] - stats['value']
        stats['in_range'] = stats.apply(lambda row: True if (row.value<=row.hdi95_high) and 
                                   (row.value>=row.hdi95_low) else False, axis=1)
    return stats

def get_predictions(density_estimator, sum_stat, theta, readable_prior, num_of_samples=1000, bins=100):
    print(f'Getting posterior from sumstat:\n{sum_stat}')
    model = sbi_post(density_estimator)
    model.generate_post(sum_stat, num_of_samples)
    post = model.get_post()
    stats = calc_stats(post, theta, readable_prior, bins=bins)
    return post, stats

def get_ensemble_predictions(ensemble_path, sum_stat, theta, readable_prior, samples_per_model=300, bins=100):
    posterior = []
    stats = []
    for model_name in os.listdir(ensemble_path):
        if '.dill' not in model_name:
            continue
        density_estimator = os.path.join(ensemble_path, model_name)
        post, model_stats = get_predictions(density_estimator, sum_stat, theta, readable_prior, samples_per_model, bins=bins)
        post['model'] = model_name
        posterior.append(post)
        model_stats['model'] = model_name
        stats.append(model_stats)
    posterior = pd.concat(posterior)
    post_stats = calc_stats(posterior, theta, readable_prior, bins=bins)
    post_stats['model'] = 'ensemble'
    stats.append(post_stats)
    stats = pd.concat(stats)
    return posterior, stats

def round_tensor(tensor, n_digits):
    return torch.round(tensor * 10**n_digits) / (10**n_digits)

def get_prior_from_params(params, readable):
    readable_prior = {k:v for k,v in params.items() if k in prior_names}
    if readable:
        return readable_prior
    prior = sbiutils.BoxUniform(low=np.array([val[0] for val in readable_prior.values()]), 
                                high=np.array([val[1] for val in readable_prior.values()]))
    return prior


def grab_long_sumstat(t):
    return t[:, :204]

def grab_man_sumstat(t):
    return torch.cat((t[:,:6],t[:,204:]), axis=1)

def grab_short_sumstat(t):
    return t[:, :6]

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

def get_long_sumstat(df, passages=[3,7,10]):
    return torch.cat((get_short_sumstat(df, passages), get_long_stats(df, passages)))

def manual_sumstat_to_dataframe(sumstat, passages=[3,7,10]):
    max_muts = 11 # so 10 maximum muts
    new_index = [(x,y,z,w) for w in range(max_muts) for z in range(max_muts)
                 for y in range(max_muts) for x in range(max_muts) if x+y<max_muts and z<=x and w<=y]
    ret = pd.DataFrame(np.array(sumstat[6:].reshape(-1,3)), columns=passages, index=new_index)
    cols_dict = {'syn_total': 0,  'non_syn_total':1, 'syn_ben':2, 'non_syn_ben':3}
    for k, v in cols_dict.items():
        ret[k] = ret.index.map(lambda x: x[v])
    ret['syn_non_ben'] = ret['syn_total'] - ret['syn_ben']
    ret['non_syn_non_ben'] = ret['non_syn_total'] - ret['non_syn_ben']
    ret = ret.set_index(ret.apply(lambda row: (int(row.syn_non_ben), int(row.non_syn_non_ben), 
                                       int(row.syn_ben), int(row.non_syn_ben)), axis=1))
    return ret

def manual_to_long_sumstat(man_sumstat):
    long_sumstat_length_per_passage = 68
    return man_sumstat.reshape(-1,3)[:long_sumstat_length_per_passage,:].flatten()

def simulate_from_post(post, num_of_samples, syn_prob):
    from simulator import simulate  # our latest simulator
    params_list = []
    syn_data = []
    i = 1
    for row in post.sample(num_of_samples).iterrows():
        print(f"Simulating sample {i}/{num_of_samples} ...", end='\r')
        params = row[1]
        params_list.append(params)
        datum = simulate(params, return_data=True, syn_prob=syn_prob)
        syn_data.append(datum)
        i += 1
    print(f"Done simulating {num_of_samples} samples!", end='\r')
    return syn_data, params_list


def verify_sumstat(sumstat):
    if sumstat!='long' and sumstat!='short' and sumstat!='man':
        raise Exception(f'sumstat should be one of [short, long, man] not {sumstat}!')

def assign_embedding_net(sumstat):
    if sumstat=='long':
        embed=True
        embedding_net = nn.Sequential(nn.Linear(204, 128), 
                        nn.ReLU(),
                        nn.Linear(128, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16))
    elif sumstat=='man':
        embed=True
        embedding_net = nn.Sequential(nn.Linear(3009, 512), 
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128, 16))
    elif sumstat=='short':
        embed=False
        embedding_net = None
    return embed, embedding_net
