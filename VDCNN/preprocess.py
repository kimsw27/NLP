import os
import pandas as pd
from sklearn.model_selection import train_test_split

# train / valid
train_file_path =  './data/ratings_train.txt'
train_df = pd.read_csv(train_file_path, sep='\t').loc[:, ['document', 'label']]
train_df = train_df[~train_df.document.isna()]
train, valid = train_test_split(train_df, test_size=0.2, random_state=1)
train.to_csv( './data/train.txt', sep='\t', index=False)
valid.to_csv( './data/valid.txt', sep='\t', index=False)

# test
test_file_path =  './data/ratings_test.txt'
test_df = pd.read_csv(test_file_path, sep='\t').loc[:, ['document', 'label']]
test_df = test_df[~test_df.document.isna()]
test_df.to_csv('./data/test.txt',  sep='\t', index=False)