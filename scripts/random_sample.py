import pandas as pd
import numpy as np

# put your csv numbers here
# also put all .csv files that you downloaded
# inside the /corona-tweets folder
csv_files = [170, 177, 184, 191, 198, 205] 
sample_size = 20000 # specify your sample size

# this is for creating ready_corona_tweets.csv
# if you already done this just set this to False
ready_dataset = True 

for file in csv_files:
    if ready_dataset:
        print("Creating the ready dataset for "+str(file)+".csv")

        dataframe = pd.read_csv(
            "./corona-tweets/corona_tweets_"+str(file)+".csv", header=None)
        dataframe = dataframe[0]
        dataframe.to_csv(
            "./ready-corona-tweets/ready_corona_tweets_"+str(file)+".csv", index=False, header=None)

    print("Creating random sample for "+str(file)+".csv")

    dataframe = pd.read_csv(
        "./ready-corona-tweets/ready_corona_tweets_"+str(file)+".csv", header=None)
    sample_indices = np.random.choice(
        dataframe.size, sample_size, replace=False)
    sample = np.take(dataframe[0], sample_indices)

    sample.to_csv(
        "./tweet-id-sampled/tweets-"+str(file)+".csv", index=False, header=None)

