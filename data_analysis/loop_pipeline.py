import argparse
import logging
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
from plotting import set_plots_size_params


def convert_fastq_to_fasta(output_dir, fastq_file):
    reads_fasta_file_name = os.path.basename(fastq_file).replace('fastq', 'fasta')
    reads_fasta_file_path = os.path.join(output_dir, reads_fasta_file_name)
    with open(fastq_file, "r") as input_handle:
        with open(reads_fasta_file_path, "w") as output_handle:
            SeqIO.convert(input_handle, "fastq", output_handle, "fasta")
    return reads_fasta_file_path


def fastq_to_df(input_fastq):
    records = []
    with open(input_fastq, 'r') as handle:
        for record in SeqIO.parse(handle, "fastq"):
            records.append({'read_id': record.id, 'read_len': len(record.seq)})
    return pd.DataFrame.from_records(records)


def run_blast(blast_params, output_dir, fastq_path, reference_file):
    fasta_file = convert_fastq_to_fasta(fastq_file=fastq_path, output_dir=output_dir)
    blast_output_file = fasta_file + ".blast"
    bp = blast_params.split(" ")
    outfmt = "6 qseqid sseqid qstart qend qstrand sstart send sstrand length btop qseq sseq"  # defines blast output
    blast_instance = NcbiblastnCommandline(query=fasta_file, subject=reference_file, out=blast_output_file,
                                           outfmt=outfmt, task=bp[2], dust=bp[4], evalue=bp[0],
                                           num_alignments=bp[3], perc_identity=bp[1],
                                           soft_masking=bp[5])
    stdout, stderr = blast_instance()
    return blast_output_file, stdout, stderr


def get_numbers_in_str(string):
    return list(map(int, re.split(r'\D+', string)))


def parse_mutations(row, mutations_list):
    position = row.ref_start - 1
    mutations_string = row.blast_diff
    numbers = get_numbers_in_str(mutations_string)
    i = 0
    next_number_index = 0
    mutations = []
    use_next_num = False
    while i < len(mutations_string):
        if mutations_string[i].isnumeric():
            i += 1
            use_next_num = True
        else:
            if use_next_num:
                position = position + numbers[next_number_index]
                next_number_index += 1
            if mutations_string[i + 1] == '-':  # insertion
                position += 0.001
            else:
                position = int(position) + 1
            mutation = mutations_string[i:i + 2][::-1]  # blast's top reads them the other way around..
            mutations.append({'mutation': mutation, 'position': round(position, 3), 'read_id': row.read_id})
            i += 2
            if not mutations_string[i].isnumeric():  # next is a mutation
                use_next_num = False
            else:
                use_next_num = True
    mutations_list += mutations
    return


def _assign_transition_type(t):
    transitions = ['AG', 'GA', 'TC', 'CT']
    transversions = ['AC', 'TG', 'TA', 'AT', 'GC', 'CG', 'CA', 'GT']
    if not isinstance(t, str):
        return 'err'
    if t in transitions:
        return 'ts'
    elif t[1] == '-':
        return 'del'
    elif t[0] == '-':
        return 'ins'
    elif t in transversions:
        return 'tv'
    else:
        return 'err'


def get_mutations(data):
    mutations_list = []
    data.apply(lambda row: parse_mutations(row, mutations_list), axis=1)
    mutations = pd.DataFrame.from_records(mutations_list)
    mutations['type'] = mutations.mutation.map(_assign_transition_type)
    mutations['mut_name'] = mutations['position'].astype(int).astype(str) + mutations['mutation']
    return mutations


def count_mutations(string):
    mutations = [x for x in string if not x.isnumeric()]
    return len(mutations)/2


def parse_blast(blast_output_file, min_alignment_len):
    df = pd.read_table(blast_output_file,
                       names=["read_id", "ref_id", "read_start", 'read_end', 'ref_start',
                              'ref_end', 'plus_or_minus', 'aln_len', 'blast_diff', 'read_seq', 'ref_seq'])
    df['mut_num'] = df['blast_diff'].map(count_mutations)
    df[f'too_short'] = 0
    df.loc[df.aln_len<min_alignment_len, f'too_short'] = 1
    df['aligned_more_than_once'] = 0
    df.loc[df.read_id.duplicated(keep=False), 'aligned_more_than_once'] = 1
    df['aligned_to_minus'] = 0
    df.loc[df.plus_or_minus == 'minus', 'aligned_to_minus'] = 1
    df['passed'] = (1-df['aligned_more_than_once']) * (1-df['aligned_to_minus']) * (1-df['too_short'])
    return df


def get_freqs(mutations, reads):
    freqs = pd.DataFrame(mutations.groupby(['mut_name']).read_id.nunique())
    freqs['freq'] = freqs['read_id'] / reads.read_id.nunique()
    freqs = freqs.reset_index().rename(columns={'read_id': 'read_count'})
    freqs['ref_pos'] = freqs['mut_name'].str[:-2]
    freqs['transition'] = freqs['mut_name'].str[-2:]
    return freqs


def aggregate_data(data):
    mutations = get_mutations(data)
    reads = data.join(mutations.groupby('read_id').mut_name.apply("_".join), on='read_id')
    reads.rename(columns={'mut_name': 'mutations'}, inplace=True)
    freqs = get_freqs(mutations, reads)
    return {'reads': reads, 'mutations': mutations, 'freqs': freqs}


def get_data(blast_output_file, fastq_path, min_alignment_len):
    blast_data = parse_blast(blast_output_file, min_alignment_len=min_alignment_len)
    fastqs_data = fastq_to_df(fastq_path)
    data = blast_data.merge(fastqs_data, on=['read_id'])
    ignored_reads = data[data.passed == 0]
    good_data = data[data.passed == 1][['read_id', 'read_start', 'read_end', 'ref_start', 'ref_end',
                                        'aln_len', 'read_len', 'blast_diff', 'read_seq', 'ref_seq', 'mut_num']].copy()
    good_ags = aggregate_data(good_data)
    full_ags = aggregate_data(data)
    return good_ags, ignored_reads, full_ags


def genomes_pieplot(reads, ax):
    pie_data = reads.mutations.value_counts(normalize=True, dropna=False).reset_index().fillna('reference')
    pie_data.loc[9:, 'index'] = 'other'
    pie_data = pie_data.groupby('index').sum().sort_values(by='mutations', ascending=False).reset_index()
    pie_data['legend'] = pie_data['index'] + ' - ' + (round(100 * pie_data['mutations'], 1)).astype(str) + '%'
    ax.pie(pie_data['mutations'], autopct='%1.1f%%')
    ax.legend(pie_data['legend'], loc=(1, 0.4))


def data_barplot(reads, ignored_data, ax):
    ignored_data.rename(columns={'shorter_than_3500': 'too_short'}, inplace=True)
    reads['passed'] = 1
    cols = ['too_short', 'aligned_more_than_once', 'aligned_to_minus']
    for col in cols:
        reads[col] = 0
    data = pd.concat([ignored_data, reads])
    ax.bar(['passed'] + cols, data[['passed'] + cols].mean())
    ax.set_ylabel('% of reads')


def wrangle_heatmap_data(data, bin_size):
    heatmap_data = data[data.groupby('read_id').ref_start.transform('count') == 1].copy()
    heatmap_data['start_ref_rounded'] = (heatmap_data.ref_start // bin_size) * bin_size
    heatmap_data['end_ref_rounded'] = (heatmap_data.ref_end // bin_size) * bin_size
    heatmap_data = heatmap_data.groupby(['start_ref_rounded', 'end_ref_rounded']).read_id.count().reset_index()
    min_alignments = min(heatmap_data['read_id'])
    max_alignments = max(heatmap_data['read_id'])
    return heatmap_data, min_alignments, max_alignments


def alignment_heatmap(ignored_reads, axes, bin_size=100):
    plus_reads = ignored_reads[(ignored_reads.plus_or_minus == 'plus')]
    plus_data, plus_min_alignments, plus_max_alignments = wrangle_heatmap_data(plus_reads, bin_size)
    minus_reads = ignored_reads[(ignored_reads.plus_or_minus == 'minus')]
    minus_data, minus_min_alignments, minus_max_alignments = wrangle_heatmap_data(minus_reads, bin_size)
    vmin = min(plus_min_alignments, minus_min_alignments)
    vmax = max(plus_max_alignments, minus_max_alignments)
    sns.heatmap(plus_data.pivot(index='end_ref_rounded', columns='start_ref_rounded', values='read_id').fillna(0), ax=axes['7'], cmap='coolwarm', vmin=vmin, vmax=vmax, cbar=False)
    sns.heatmap(minus_data.pivot(index='start_ref_rounded', columns='end_ref_rounded', values='read_id').fillna(0), ax=axes['8'], cmap='coolwarm', vmin=vmin, vmax=vmax, cbar_ax=axes['9'],  cbar_kws={'label': 'number of alignments'})
    axes['7'].invert_yaxis()
    axes['8'].invert_yaxis()
    axes['7'].set_xlabel('alignment start')
    axes['8'].set_xlabel('alignment start')
    axes['7'].set_ylabel('alignment end')
    axes['8'].set_ylabel('alignment end')
    axes['8'].set_yticks([])
    axes['7'].set_title('plus alignments')
    axes['8'].set_title('minus alignments')


def coverage_hist(data, ax):
    data['real_start'] = data['ref_end']
    data['real_end'] = data['ref_start']
    data.loc[data.plus_or_minus == 'plus', 'real_start'] = data.loc[data.plus_or_minus=='plus', 'ref_start']
    data.loc[data.plus_or_minus == 'plus', 'real_end'] = data.loc[data.plus_or_minus=='plus', 'ref_end']
    asd = []
    data.apply(lambda row: asd.extend(list(range(row.real_start, row.real_end, 20))), axis=1)
    plt.hist(asd, bins=30, ax=ax)


def mutation_types_barplot(mutations, ax):
    type_counts = mutations.groupby('type').read_id.value_counts()
    type_counts.name = 'count'
    type_counts = type_counts.reset_index()
    for mutype in mutations.type.unique():
        num_of_zeros = mutations.read_id.nunique() - len(type_counts[type_counts.type == mutype])
        tmp = pd.DataFrame({'type': mutype, 'read_id': 'tmp', 'count': np.zeros(num_of_zeros)})
        type_counts = pd.concat([type_counts, tmp])
    sns.barplot(data=type_counts, x='type', y='count', ax=ax)


def plot_data(reads, mutations, ignored_reads, output_dir):
    set_plots_size_params(20)
    ax_dict = plt.figure(constrained_layout=False, figsize=(60, 20)).subplot_mosaic(
                         """
                         1122334
                         5566789
                         """,
                         gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 0.08]})
    plt.suptitle("Pipeline Statistics", fontsize=18)
    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    full_data = pd.concat([reads, ignored_reads])
    ax = ax_dict['5']
    ax.set_title('Data Overview')
    data_barplot(reads, ignored_reads, ax=ax)
    ax = ax_dict['6']
    ax.set_title('Alignment Length Histogram')
    sns.distplot(full_data.aln_len, bins=200, ax=ax, kde=False)
    ax = ax_dict['3']
    ax.set_title('Common Genotypes')
    genomes_pieplot(reads, ax=ax)
    ax = ax_dict['2']
    ax.set_title('Mutations per Genome (excluding 0.1% outliers!)')
    sns.distplot(reads[reads.mut_num < reads.mut_num.quantile(0.999)].mut_num, ax=ax, kde=False)
    ax = ax_dict['1']
    ax.set_title('Mutation Type per Genome')
    mutation_types_barplot(mutations, ax=ax)
    alignment_heatmap(ignored_reads=ignored_reads, axes=ax_dict)
    ax_dict['4'].set_axis_off()
    plt.savefig(os.path.join(output_dir, "pipeline_figures.png"))


def main(fastq_path, reference_file, output_dir, blast_params, min_alignment_len):
    unfiltered_dir = os.path.join(output_dir, 'unfiltered')
    os.makedirs(unfiltered_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO,
                        filename=os.path.join(output_dir, '.log'))
    logging.warning(f"Started loop pipeline with params: {args}")
    try:
        blast_output_file, stdout, stderr = run_blast(blast_params, output_dir, fastq_path, reference_file)
        good_ags, ignored_reads, full_ags = get_data(blast_output_file, fastq_path, min_alignment_len)
        plot_data(good_ags['reads'], good_ags['mutations'], ignored_reads, output_dir)
        good_ags['reads'].to_csv(os.path.join(output_dir, 'reads.tsv'), sep='\t', index=False)
        good_ags['mutations'].to_csv(os.path.join(output_dir, 'mutations.tsv'), sep='\t', index=False)
        good_ags['freqs'].to_csv(os.path.join(output_dir, 'freqs.tsv'), sep='\t', index=False)
        ignored_reads.to_csv(os.path.join(output_dir, 'ignored_reads.tsv'), sep='\t', index=False)
        full_ags['reads'].to_csv(os.path.join(unfiltered_dir, 'unfiltered_reads.tsv'), sep='\t', index=False)
        full_ags['mutations'].to_csv(os.path.join(unfiltered_dir, 'unfiltered_mutations.tsv'), sep='\t', index=False)
        full_ags['freqs'].to_csv(os.path.join(unfiltered_dir, 'unfiltered_freqs.tsv'), sep='\t', index=False)
        logging.info(f"Ended successfully! Files should be in {output_dir}")
    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--fastq_path", required=True, help="Path to fastq file containing sequence data")
    parser.add_argument("-o", "--output_dir", help="A directory for output files.")
    parser.add_argument("-r", "--reference_file", required=True,
                        help="Full path to reference file (including extension) of type fasta to align against.")
    parser.add_argument("-b", "--blast_params", required=False, default="1e-07 0.85 blastn 1000000 no F",
                        help="space separated blast params in the following order: evalue perc_identity task "
                             "num_alignments dust soft_masking. (default: 1e-07 0.85 blastn 1000000 no F)")
    parser.add_argument("-m", "--min_alignment_len", required=False, default=3500, type=int,
                        help="Alignments shorter than this many bases will be put in ignored reads")
    args = vars(parser.parse_args())
    main(fastq_path=args['fastq_path'], reference_file=args['reference_file'], output_dir=args['output_dir'],
         blast_params=args['blast_params'], min_alignment_len=args['min_alignment_len'])
