import numpy as np
import pandas as pd
import re
import glob

path = '../'
all_files = glob.glob(path + "/swcwang-dataset/*.csv")

li = []

for filename in all_files:
    frame = pd.read_csv(filename,  encoding='unicode_escape',
                        skipinitialspace=True)
    li.append(frame)

df = pd.concat(li, axis=0, ignore_index=True)

all_df = df['tweet_processed']
all_df.to_csv(path+"swcwang-processed-dataset/tweets_combined.csv",
              index=True, header=['tweet'])

df_target_1 = df[df['target'] == 1]
df_target_1 = df_target_1['tweet_processed']
df_target_1.to_csv(path+"swcwang-processed-dataset/tweets_combined_target_1.csv",
                   index=True, header=['tweet'])
