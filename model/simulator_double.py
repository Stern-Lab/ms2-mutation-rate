from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")

def simulate(params, syn_prob=0.278, passages=10, pop_size=10**9, return_data=False):       
    # dill requires imports to be made within the simulator function
    import numpy as np
    from .evolutionary_model import simulate_p0, simulate_next_passage, wrangle_data, get_total_sumstat
    from .evolutionary_model import get_mutations

    try:
        mutation_rate = 10 ** params[0]
        w_syn = params[1]
        w_non_syn = params[2]
        w_ada = params[3]
        p_ada_syn = params[4]
        p_ada_non_syn = params[5]
        p0_syn = params[6]
        p0_non_syn = params[7]
        w_penalty = params[8]
        epistasis_boost = params[9]
        
        fitness_effects = np.array([w_syn**w_penalty, w_non_syn**w_penalty, w_syn, w_non_syn, w_ada])
        passage = dict()
        passage[0] = simulate_p0(p0_syn, p0_non_syn, pop_size)
        mutations = get_mutations(mutation_rate, syn_prob, p_ada_syn, p_ada_non_syn, 1/(100*pop_size))
        mutations = {(0,0)+mut :freq for mut, freq in mutations.items()}
        for i in range(passages):
            mid_passage = simulate_next_passage(fitness_effects, passage[i], mutations, pop_size,
                                                 epistasis_boost)
            passage[i+1] = simulate_next_passage(fitness_effects, mid_passage, mutations, pop_size,
                                                 epistasis_boost)
        data = wrangle_data(passage)
        if not return_data:
            data = get_total_sumstat(data)
    except Exception as e:
        raise Exception(f"Exception: '{e}' occured with params: {params}")
    return data
