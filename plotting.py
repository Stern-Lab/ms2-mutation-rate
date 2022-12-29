import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from model.simulator import simulate
from utils import get_short_sumstat, manual_sumstat_to_dataframe, simulate_from_post


def set_plots_size_params(size):
    # Adapted from https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    bigger = size * 1.2
    slightly_bigger = size * 1.1
    plt.rc('font', size=size)                        # controls default text sizes
    plt.rc('axes', titlesize=bigger)                 # fontsize of the axes title
    plt.rc('axes', labelsize=slightly_bigger)        # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)                  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)                  # fontsize of the tick labels
    plt.rc('legend', fontsize=size)                  # legend fontsize
    plt.rc('figure', figsize=(0.75*size, size/2))    # size of figure

    
def short_sumstats_plot(syn_data, man_sumstat, colors=['C1','C0'], alpha=0.3, label_suffix=''):
    legend = True
    passages = list(range(11))
    for syn_datum in syn_data:
        sumstat = get_short_sumstat(syn_datum, passages)
        non_syns = [sumstat[x] for x in range(len(sumstat)) if x%2==1]
        syns = [sumstat[x] for x in range(len(sumstat)) if x%2==0]
        plt.plot(passages, non_syns, 'o-', label='simulated non-syn'+label_suffix if legend else '_nolegend', 
                 color=colors[0], alpha=alpha)
        plt.plot(passages, syns,  'o-', label='simulated syn'+label_suffix if legend else '_nolegend', 
                 color=colors[1], alpha=alpha)
        legend = False
    if man_sumstat is not None:
        short_sumstat = man_sumstat[:6]
        passages = [3,7,10]
        non_syns = [short_sumstat[x] for x in range(len(short_sumstat)) if x%2==1]
        syns = [short_sumstat[x] for x in range(len(short_sumstat)) if x%2==0]
        plt.plot(passages, non_syns, '^-', label='empirical non-syn'+label_suffix, color=colors[0], 
                 linewidth=3, markersize=15)
        plt.plot(passages, syns, '^-', label='empirical syn'+label_suffix, color=colors[1], 
                 linewidth=3,  markersize=15)
    plt.xlabel('Passage')
    plt.ylabel('Avg Mutations Per Genome')
    plt.title(f"Simulations from Posterior: Mutations Per Genotype")
    # remove alpha from first 2 legend handles
    h, l = plt.gca().get_legend_handles_labels()
    h[:2] = [plt.Line2D([], [], color=colors[i], linestyle='solid', markersize=8, marker='.') for i in range(2)]
    plt.legend(h,l)
    
def man_sumstats_plot(syn_data, empiric_data_df, colors=None, alpha=0.3):
    legend = True
    passages = list(range(11))
    if colors is None:
        colors = sns.color_palette(None)
    for datum in syn_data:
        plt.plot(passages, datum.loc[0, passages], 'x-', label='simulated WT' if legend else '_nolegend', 
                 color=colors[0], alpha=alpha, markerfacecolor='white', markersize=8)
        syn_bens, non_syn_bens = get_bens(datum, passages)
        plt.plot(passages, non_syn_bens, 'x-', label='simulated beneficial non-syn' if legend else '_nolegend', 
                 color=colors[1], alpha=alpha, markersize=8 )
        plt.plot(passages, syn_bens,  'x-', label='simulated beneficial syn' if legend else '_nolegend', 
                 color=colors[2], alpha=alpha, markersize=8)
        legend = False
    linestyle = 'dashdot'
    if empiric_data_df is not None:
        passages = [3,7,10]
        plt.plot(passages, empiric_data_df.iloc[0,[0,1,2]], marker='^', linestyle=linestyle, label='empirical WT', 
                 color=colors[0], linewidth=3, markersize=15, markerfacecolor='white')
        syn_bens, non_syn_bens = get_bens(empiric_data_df, passages)
        plt.plot(passages, non_syn_bens,  marker='^', linestyle=linestyle, label='empirical beneficial non-syn', 
                 color=colors[1], linewidth=3, markersize=15)
        plt.plot(passages, syn_bens,  marker='^', linestyle=linestyle, label='empirical beneficial syn', 
                 color=colors[2], linewidth=3,  markersize=15)
    plt.xlabel('Passage')
    plt.ylabel('Frequency')
    plt.title(f"Simulations from Posterior: WT and Beneficial Genotypes")
    # remove alpha from first 3 legend handles
    h, l = plt.gca().get_legend_handles_labels()
    h[:3] = [plt.Line2D([], [], color=colors[i], linestyle='solid', markersize=8, marker='x') for i in range(3)]
    plt.legend(h,l)
    

def genome_freq_plot(data, ax, num_of_genomes=10, color=None):
    df = data.set_index(['syn_pri', 'non_syn_pri','syn','non_syn','syn_ben','non_syn_ben'])
    last_passage = 10
    rising_muts = df.sort_values(by=last_passage, ascending=False).head(num_of_genomes).index
    passages = list(range(last_passage+1))
    ax.plot(df[passages].loc[(0,0,0,0,0,0)], label='WT', marker='*', color=color)
    syn_data = df[df.index.get_level_values(4)>=1][passages].sum()
    non_syn_ada = df[df.index.get_level_values(5)>=1][passages].sum()
    ax.plot(syn_data.index, syn_data, label='cumulative syn beneficial', marker='*', linestyle='--', color=color)
    ax.plot(non_syn_ada.index, non_syn_ada, label='cumulative non-syn beneficial', marker='*', linestyle='--', color=color)
    for mut in rising_muts:
        if mut == (0,0,0,0,0,0):
            continue
        ax.plot(df.loc[mut, passages], label=str(mut), marker='*', color=color)
    ax.legend(bbox_to_anchor=(1,1.02))
    ax.xlabel('Passage')
    ax.ylabel('Frequency')


def plot_HDIs(post, palette='Greys', marginal_color='grey'):
    az.rcParams.update({'plot.max_subplots':90}) # necessary because we do a lot of plots
    data = post.to_dict('list')
    az.plot_pair(
        data,
        var_names=list(data.keys()),
        kind="kde",
        divergences=False,
        marginals=True,
        textsize=100,
        kde_kwargs={
            "contourf_kwargs": {"cmap": palette}, 'hdi_probs': [0.1, 0.5, 0.75, 0.95]},
        marginal_kwargs={'kind': 'hist', 'color': marginal_color},
    )

def plot_from_post(post, stats, man_sumstat, syn_prob, num_of_samples=10, plot_hdi=True, plot_short_PPC=True, 
                   plot_freqs=True, plot_long_PPC=True, simulate=simulate):
    if plot_hdi:
        print('Creating HDI KDEs...')
        plot_HDIs(post)
        plt.show()
    ret = {}
    if plot_short_PPC or plot_long_PPC:
        syn_data, params = simulate_from_post(post, num_of_samples, syn_prob)
        ret['syn_data'] = syn_data
        ret['syn_data_params'] = params
    if plot_short_PPC:
        short_sumstats_plot(syn_data, man_sumstat)
        plt.show()
    if plot_long_PPC:
        man_sumstats_plot(syn_data, manual_sumstat_to_dataframe(man_sumstat))
        plt.show()
    if plot_freqs:
        print('Creating genotype class plot...')
        mode_theta = list(stats[stats.model=='ensemble']['mode'])
        mode_data = simulate(mode_theta, return_data=True, syn_prob=syn_prob)
        genome_freq_plot(mode_data, ax=plt)
        plt.title('Genotype Classes of the Mode of the Posterior')
        plt.show()
        ret['full_mode_data'] = mode_data
    return ret

def syn_non_syn_freq_plot(data, ax, num_of_genomes=10, color=None):
    passages = list(range(11))
    df = data.groupby(['syn_mut_num', 'reg_mut_num'])[passages].sum()
    rising_muts = df.sort_values(by=10, ascending=False).head(num_of_genomes).index
    ax.plot(df[passages].loc[(0,0)], label='WT', marker='*', color=color)
    """syn_data = df[df.index.get_level_values(4)>=1][passages].sum()
    non_syn_ada = df[df.index.get_level_values(5)>=1][passages].sum()
    ax.plot(syn_data.index, syn_data, label='syn adas', marker='*', linestyle='--', color=color)
    ax.plot(non_syn_ada.index, non_syn_ada, label='non syn adas', marker='*', linestyle='--', color=color)"""
    for mut in rising_muts:
        if mut == (0,0):
            continue
        ax.plot(df.loc[mut, passages], label=str(mut), marker='*', color=color)
    ax.legend(bbox_to_anchor=(1,1.02))
    ax.xlabel('Passage')
    ax.ylabel('Frequency')
    
def compare_posts(posts, title_suffix='', colors=None, alpha=0.7):
    fig, axes = plt.subplots(4, 3, figsize=(18,15), constrained_layout=True)
    ax_dict = {r'Mutation rate (U)': (1,1), r'Syn fitness effect ($w_{s}$)': (1,2), 
               r'Non-syn fitness effect ($w_{ns}$)': (1,0), 
               r'Mean initial syn ($M_{s}$)': (0,0), r'Mean initial non-syn ($M_{ns}$)': (0,2),  
               r'Beneficial fitness effect ($w_{b}$)': (3,0), r'Initial log-fitness correlation ($\delta$)': (3,2), 
               r'Epistasis effect ($\eta$)': (3,1), r'Beneficial syn probability ($p_{bs}$)': (2,2), 
               r'Beneficial non-syn probability ($p_{bns}$)': (2,0)}
    if not colors:
        colors = sns.color_palette(None)
    title = 'Marginal Posteriors'
    title += title_suffix
    fig.suptitle(title)
    legend = True
    for col_name, ax_index in ax_dict.items():
        ax = axes[ax_index]
        col = 'err' if 'Mutation' in col_name else 'rel_err'
        for i, (name, df) in enumerate(posts.items()):
            df[col_name].hist(bins=20, label=name if legend else '_nolegend', ax=ax, alpha=alpha, color=colors[i])
        ax.set_xlabel(col_name.replace('\n', ' '))
        ax.set_ylabel('Count')
        legend = False
    fig.delaxes(axes[(0,1)])
    fig.delaxes(axes[(2,1)])
    fig.legend(loc=(0.4,0.85))

def compare_posts_across_lines(outputs_dict, colors=None, alpha=0.7):
    if 'A' in outputs_dict.keys():
        compare_posts({line: output[0] for line, output in outputs_dict.items()}, 
                      colors=colors, alpha=alpha, title_suffix=' by Replica')
    else:
        lines_dict = {1:'A', 2:'B', 3:'C'}
        compare_posts({lines_dict[line]: output[0] for line, output in outputs_dict.items()}, 
                  colors=colors, alpha=alpha, title_suffix=' by Replica')

def get_bens(data, passages):
    bens = data.groupby(['syn_ben', 'non_syn_ben'])[passages].sum()
    syn_bens = bens[bens.index.get_level_values(0)>=1][passages].sum()
    non_syn_bens = bens[bens.index.get_level_values(1)>=1][passages].sum()
    return syn_bens, non_syn_bens
