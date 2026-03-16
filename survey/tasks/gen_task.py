import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import wget
import zipfile

def visualize(X, y):
    '''
        Visualize a dataset using PCA reduction of features
        into two dimensions. Color according to the labels.
    '''
    raise Exception('Visualization not yet implemented!')

def find_file(filename, search_path):
    '''
        Method for finding a file in a subdirectory with a
        specific name. Used to find the selected feature names
        in the MADML directories.
    '''
    search_path = Path(search_path)
    for path in search_path.rglob(filename):
        return path  # returns the first match
    return None

madml_links = {
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
    parser.add_argument('--task', 
        choices=[
            'bace',
            'bandgap',
            'bear',
            'chain_gap',
            'cloud_point',
            'concrete',
            'd_max',
            'debye',
            'diffusion',
            'elastic',
            'esol',
            'exfoliation',
            'freesolv',
            'gox_rea',
            'hplc',
            'hrp_rea',
            'lip_rea',
            'lipo',
            'logp',
            'muller_brown',
            'oer',
            'oxide',
            'pcqm4mv2_small',
            'perovskite_cond',
            'perovskite_formation',
            'perovskite_h_adsorp',
            'perovskite_o_p_band',
            'perovskite_stability',
            'perovskite_work',
            'phonon_freq',
            'piezo',
            'polygas_CH4',
            'polygas_CO2',
            'polygas_H2',
            'polygas_He',
            'polygas_N2',
            'polygas_O2',
            'polymer_density',
            'polymer_mt',
            'polymer_o2',
            'polymer_tg',
            'qm9_alpha',
            'qm9_cv',
            'qm9_g298',
            'qm9_h298',
            'qm9_homo',
            'qm9_lumo',
            'qm9_mu',
            'qm9_r2',
            'qm9_u0',
            'qm9_u298',
            'qm9_zpve',
            'rc',
            'rpv',
            'semiconductor',
            'sublimation',
            'superconductivity',
            'thermal_cond_aflow',
            'thermal_cond_exp',
            'thermal_expansion',
            'toporg',
        ]
    )
    args = parser.parse_args()

    # Preparation of MADML datasets from Jacobs et al.
    if args.task in madml_links.keys():

        # Cleanly download content from figshare link, unzip.
        wget.download(madml_links[args.task])
        zip_file = [filename for filename in os.listdir('.') if '.zip' in filename][0]
        with zipfile.ZipFile(f'./{zip_file}', 'r') as zip_ref:
            zip_ref.extractall(f'.')
        os.system(f'rm {zip_file}')
        if '__MACOSX' in os.listdir('.'):
            os.system(f'rm -rf __MACOSX')
        dir_name = zip_file[:-4]
        print('')

        # # Get raw data (stored in data/ directory).
        # raw_file = [filename for filename in os.listdir(f'./{dir_name}/data/') if '.csv' in filename][0]
        # raw_data = pd.read_csv(f'./{dir_name}/data/{raw_file}') # This is just for future reference; not saved.

        if args.task != 'superconductivity':

            # Get featurized data and labels.
            X_train_path = find_file('X_train.csv', f'./{dir_name}/5fold')
            X_test_path = find_file('X_test.csv', f'./{dir_name}/5fold')
            y_train_path = find_file('y_train.csv', f'./{dir_name}/5fold')
            y_test_path = find_file('y_test.csv', f'./{dir_name}/5fold')
            X_train = pd.read_csv(X_train_path).to_numpy()
            X_test = pd.read_csv(X_test_path).to_numpy()
            y_train = pd.read_csv(y_train_path).to_numpy().reshape(-1)
            y_test = pd.read_csv(y_test_path).to_numpy().reshape(-1)
            X = np.vstack((X_train, X_test))
            y = np.hstack((y_train, y_test))

            # Get feature names.
            feature_names_path = find_file(f'selected_features.txt', f'./{dir_name}/5fold')
            feature_names_file = open(feature_names_path, 'r').readlines()
            feature_names = [line.strip() for line in feature_names_file]

        # The superconductivity dataset was not stored appropriately...
        else:

            X_train_path = find_file('X_train.csv', f'./{dir_name}/5fold')
            X_test_path = find_file('X_test.csv', f'./{dir_name}/5fold')
            y_train_path = find_file('y_train.csv', f'./{dir_name}/5fold')
            y_test_path = find_file('y_test.csv', f'./{dir_name}/5fold')
            X_train = pd.read_csv(X_train_path)
            X_test = pd.read_csv(X_test_path).to_numpy()
            y_train = pd.read_csv(y_train_path).to_numpy().reshape(-1)
            y_test = pd.read_csv(y_test_path).to_numpy().reshape(-1)

            feature_names = X_train.columns.tolist()
            X_train = X_train.to_numpy()
            X = np.vstack((X_train, X_test))
            y = np.hstack((y_train, y_test))

            subsampled_idx = [i for i in range(X.shape[0])]
            np.random.shuffle(subsampled_idx)
            subsampled_idx = subsampled_idx[0:50000]
            X = X[subsampled_idx]
            y = y[subsampled_idx]

        # Save a labeled .csv file of the final dataset.
        data = np.hstack((X, y.reshape(-1,1)))
        dataset = pd.DataFrame(data)
        feature_names.append('target')
        dataset.columns = feature_names
        dataset.to_csv(f'./{args.task}.csv')
        print(f'Shape of {args.task}: {dataset.shape}')

        # Remove directory.
        os.system(f'rm -rf {dir_name}')

    elif args.task == 'hplc':

        df = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/refs/heads/main/src/olympus/datasets/dataset_hplc/data.csv')
        df.columns = ['sample_loop', 'additional_volume', 'tubing_volume', 'sample_flow', 'push_speed', 'wait_time', 'target']
        df.to_csv(f'./{args.task}.csv')
        print(f'Shape of {args.task}: {df.shape}')

    elif args.task == 'oer':

        df1 = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_oer_plate_3496/data.csv', header=None)
        df2 = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_oer_plate_3851/data.csv', header=None)
        df3 = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_oer_plate_3860/data.csv', header=None)
        df4 = pd.read_csv('https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/src/olympus/datasets/dataset_oer_plate_4098/data.csv', header=None)

        df = pd.concat([df1, df2, df3, df4], ignore_index=True)
        df = df.groupby([0, 1, 2, 3, 4, 5]).mean().reset_index()
        df.columns = ['ni_load', 'fe_load', 'co_load', 'mn_load', 'ce_load', 'la_load', 'target']
        df.to_csv(f'./{args.task}.csv')
        print(f'Shape of {args.task}: {df.shape}')

    elif args.task not in madml_links:
        raise Exception('Task taken from previously constructed resource...see SI Section 1 for reference.') 
