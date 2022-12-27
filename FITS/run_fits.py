import sys
import os
import subprocess
import numpy as np
import pandas as pd


def create_fits_input_data(data, tmp_path):
    outfile = f'{tmp_path}/fits_tmp_input_data.tsv'
    fits = dict()
    fits['mut'] = pd.DataFrame(data={'freq': data, 'gen': [3,7,10]})
    fits['wt'] = pd.DataFrame(data={'freq': 1-data, 'gen': [3,7,10]})
    fits['mut']['allele'] = 1
    fits['wt']['allele'] = 0
    fits_input = pd.concat(fits.values()).reset_index()
    fits_input=fits_input[['gen','allele','freq']]
    fits_input.to_csv(outfile, index=False, sep='\t')
    return outfile
    
def run_FITS(data, mut_name, tmp_path):
    """FITS params need to be edited manually"""
    input_data_file = create_fits_input_data(data, create_fits_input_data)
    run_type = 'mutation'
    cmd = f"./fits1.3.3 -{run_type} {params_file} {input_data_file} {tmp_path}/fits_post_{mut_name}.tsv {tmp_path}/fits_sum_{mut_name}.txt"
    subprocess.run(cmd.split())

    
def main(FITS_input, output_path, FITS_path, params_file='params.txt', tmp_path='tmp/'): 
    
    loop_syns_with_zeros = pd.read_table(FITS_input, index_col=0).sort_values('passage')
    os.makedirs(tmp_path, exist_ok=True)
    os.chdir(FITS_path)

    # Run FITS
    for line in loop_syns_with_zeros.line.unique():
        tmp = loop_syns_with_zeros[loop_syns_with_zeros.line==line]
        for mut_name in tmp.mut_name.unique():
            run_FITS(tmp[tmp.mut_name==mut_name].freq, f'{mut_name}_line{line}', tmp_path)

    # Aggregate outputs
    fits_post = []
    for post_name in os.listdir(tmp_path):
        if 'post' not in post_name:
            continue
        try:
            post = pd.read_table(os.path.join(tmp_path,post_name))
        except:
            print(f'DAMN! couldnt load {post_name}')
        post['source'] = post_name
        fits_post.append(post)
    fits_post = pd.concat(fits_post)
    fits_post['U'] = fits_post['allele0_1']*3500
    fits_post['log_U'] = np.log10(fits_post['U'])
    fits_post['line'] = fits_post['source'].str[-5].astype(int)
    fits_post['mut_name'] = fits_post['source'].map(lambda s: s.split('_')[2])
    fits_post['position'] = fits_post['mut_name'].str[:-2].astype(int)
    fits_post['read_base'] = fits_post['mut_name'].str[-1]
    fits_post.to_csv('output_path', sep='\t')
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
