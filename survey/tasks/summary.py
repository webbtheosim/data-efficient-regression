import os
import pandas as pd

for task_name in os.listdir('.'):
    if '.csv' in task_name:
        dataset = pd.read_csv(f'./{task_name}', index_col=0)
        print(f'Shape of {task_name[:-4]}: {dataset.shape}')
