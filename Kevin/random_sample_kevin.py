import pandas as pd
import numpy as np
import pathlib
import sys

# 1. List of the name of directories to be created
directories = [
    'ready-corona-tweets',
    'tweet-id-sampled'
]

# 2. Create directories as needed, if not exists
current_path = "E:/Github/comp4710_project/Kevin/" # Path to the random_sample.py script

for dir in directories: 
    try:
        pathlib.Path(current_path + dir).mkdir(parents=True, exist_ok=False)
        print("Created " + current_path + dir)
    except FileExistsError as e:
        print(current_path + dir + " folder already exists.")

# 3. CSV file processing

# put your csv numbers here
# also put all .csv files that you downloaded
# inside the /corona-tweets folder
csv_files = [1, 2, 3, 10, 16, 23, 30, 37] 
sample_size = 20000 # specify your sample size

# this is for creating ready_corona_tweets.csv
# if you already done this just set this to False
ready_dataset = True 

for file in csv_files:
    # if the number is under 10, add extra 0 in front of the number.
    if ready_dataset:
        if file < 10:
            file = '0' + str(file)
        else:
            file = str(file)

        print("Creating the ready dataset for "+file+".csv")

        dataframe = pd.read_csv("./corona-tweets/corona_tweets_"+file+".csv", header=None)
        dataframe = dataframe[0]
        dataframe.to_csv("./ready-corona-tweets/ready_corona_tweets_"+file+".csv", index=False, header=None)

    print("Creating random sample for "+file+".csv")

    dataframe = pd.read_csv("./ready-corona-tweets/ready_corona_tweets_"+file+".csv", header=None)
    sample_indices = np.random.choice(dataframe.size, sample_size, replace=False)
    sample = np.take(dataframe[0], sample_indices)

    sample.to_csv("./tweet-id-sampled/tweets-"+file+".csv", index=False, header=None)

