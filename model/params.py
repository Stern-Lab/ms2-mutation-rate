readable_prior = {'mu' : (-4,0.3),
                  'w_syn': (0.01,1),
                  'w_non_syn': (0.01,1),
                  'w_ada': (1,3),
                  'p_ada_syn': (0, 0.01),
                  'p_ada_non_syn': (0, 0.01),
                  'p0_syn': (0.4,0.6),
                  'p0_non_syn': (0.7,0.9),
                  'w_penalty': (0,2),
                  'epistasis_boost': (-1,3)}
sample_sizes_A = [1066, 3351, 1112]
sample_sizes_B = [1919, 2038, 889]
sample_sizes_C = [1945, 1011, 2342]
syn_prob_coding_only=0.278
passages=10 
pop_size=10**9
genome_length=3560