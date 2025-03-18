import os
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':

    # final data directory
    data_dir = './resnet_data'

    # read CSV file
    df = pd.read_csv('./output.csv')


    # Read data
    mp = dict()
    for index, row in df.iterrows():
        if row['city'] not in mp:
            mp[row['city']] = []
        mp[row['city']].append(row['picture_name'])
    
    # Create directories
    os.makedirs(data_dir)
    os.makedirs(os.path.join(data_dir, 'train'))
    os.makedirs(os.path.join(data_dir, 'val'))
    os.makedirs(os.path.join(data_dir, 'test'))

    # Create directories for each city
    for key in mp.keys():
        os.makedirs(os.path.join(data_dir, 'train', key))
        os.makedirs(os.path.join(data_dir, 'val', key))
        os.makedirs(os.path.join(data_dir, 'test', key))

    # Split data
    for city in mp:
        train, test = train_test_split(mp[city], test_size=0.2, random_state=114514)
        train, val = train_test_split(train, test_size=0.15, random_state=114514)
        for pic in train:
            os.system(f'cp "./data/{pic}" "{data_dir}/train/{city}/{pic}"')
        for pic in val:
            os.system(f'cp "./data/{pic}" "{data_dir}/val/{city}/{pic}"')
        for pic in test:
            os.system(f'cp "./data/{pic}" "{data_dir}/test/{city}/{pic}"')
