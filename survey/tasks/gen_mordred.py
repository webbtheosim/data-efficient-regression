from mordred import Calculator, descriptors
import numpy as np
import os
import pandas as pd
from rdkit import Chem

def smiles_to_mordred(smiles):
    '''
        Converts a list of SMILES strings to a pandas dataframe of
        Mordred descriptors. All constant descriptors are removed.
    '''
    calc = Calculator(descriptors, ignore_3D=False)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    df = calc.pandas(mols)
    df = df.dropna(axis=1)
    df = df.loc[:, df.nunique() != 1]
    return df

if __name__ == '__main__':

    for filename in os.listdir('./smiles'):
        print(f'Working on file: {filename}')
        
        if 'polygas' not in filename:
            df = pd.read_csv(f'./smiles/{filename}', index_col=0)
        else:
            df = pd.read_csv(f'./smiles/{filename}')
        smiles = df.iloc[:,0].to_list()
        labels = df.iloc[:,1]

        if 'qm9' in filename:
            print('Down-selecting QM9 dataset.')
            y = labels.to_numpy().reshape(-1)
            sorted_idx = np.argsort(y).reshape(-1)
            chosen_idx = sorted_idx[::10]
            smiles = [smiles[idx] for idx in chosen_idx]
            labels = labels.iloc[chosen_idx]

        desc = smiles_to_mordred(smiles)
        df = pd.concat([desc, labels], axis=1)

        print(df.head(10))

        df.to_csv(f'./mordred/{filename}')
