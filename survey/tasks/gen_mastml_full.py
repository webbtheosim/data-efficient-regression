import os
import pandas as pd
import wget
import zipfile

mastml_links = {
    'bandgap': 'https://figshare.com/ndownloader/files/47177467',
    'concrete': 'https://figshare.com/ndownloader/files/47177446',
    'debye': 'https://figshare.com/ndownloader/files/47177131',
    'dielectric': 'https://figshare.com/ndownloader/files/47183428',
    'diffusion': 'https://figshare.com/ndownloader/files/47177416',
    'perovskite_gap': 'https://figshare.com/ndownloader/files/47177440',
    'elastic': 'https://figshare.com/ndownloader/files/47177461',
    'exfoliation': 'https://figshare.com/ndownloader/files/47177425',
    'd_max': 'https://figshare.com/ndownloader/files/47177431',
    'rc': 'https://figshare.com/ndownloader/files/47177437',
    'oxide': 'https://figshare.com/ndownloader/files/47177212',
    'perovskite_cond': 'https://figshare.com/ndownloader/files/47183395',
    'perovskite_formation': 'https://figshare.com/ndownloader/files/47177473',
    'perovskite_h_adsorp': 'https://figshare.com/ndownloader/files/47177428',
    'perovskite_o_p_band': 'https://figshare.com/ndownloader/files/47177458',
    'perovskite_stability': 'https://figshare.com/ndownloader/files/47183401',
    'perovskite_work': 'https://figshare.com/ndownloader/files/47177434',
    'phonon_freq': 'https://figshare.com/ndownloader/files/47177449',
    'piezo': 'https://figshare.com/ndownloader/files/47177455',
    'rpv': 'https://figshare.com/ndownloader/files/47183404',
    'semiconductor': 'https://figshare.com/ndownloader/files/47177443',
    'superconductivity': 'https://figshare.com/ndownloader/files/47177470',
    'thermal_cond_exp': 'https://figshare.com/ndownloader/files/47177464',
    'thermal_cond_aflow': 'https://figshare.com/ndownloader/files/47177479',
    'thermal_expansion': 'https://figshare.com/ndownloader/files/47177476',
}

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='superconductivity')
    args = parser.parse_args()

    wget.download(mastml_links[args.task])
    zip_file = [filename for filename in os.listdir('.') if '.zip' in filename][0]
    with zipfile.ZipFile(f'./{zip_file}', 'r') as zip_ref:
        zip_ref.extractall(f'.')
    os.system(f'rm {zip_file}')
    if '__MACOSX' in os.listdir('.'):
        os.system(f'rm -rf __MACOSX')
    dir_name = zip_file[:-4]
    print('')

    run_file = [filename for filename in os.listdir(f'./{dir_name}/run_script')][0]
    run_file_lines = open(f'./{dir_name}/run_script/{run_file}', 'r').readlines()

    for line_idx, line in enumerate(run_file_lines):
        if 'ElementalFeatureGenerator' in line:
            print('Found preprocessor...')
            write_line = line_idx
        if 'd = LocalDatasets(file_path=' in line:
            run_file_lines[line_idx] = line.replace('mastml', f'./{dir_name}')
    write_line += 3
    run_file_lines.insert(write_line, 'raise Exception(\'!\')\n')
    run_file_lines.insert(write_line, f'df.to_csv(\'./mastml/{args.task}.csv\')\n')
    run_file_lines.insert(write_line, f'df = pd.concat([X, y], axis=1)\n')
    run_file_lines.insert(write_line, f'import pandas as pd\n')

    with open('temp.py', 'w') as handle:
        handle.writelines(run_file_lines)

    os.system('python temp.py')

    os.system(f'rm -r {dir_name}')
    os.system(f'rm temp.py')
