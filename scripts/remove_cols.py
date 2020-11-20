import pandas as pd
import numpy as np
import pathlib
import sys
import string
import os
import shutil



#Put your current directory
current_path = "/Users/Apple/Documents/COMP4710_Project/" 

#Creates a new directory 
newDir = "preprocessed_data"
shutil.rmtree(current_path + newDir, ignore_errors=True)
os.makedirs(current_path + newDir)

#Put csv numbers
csv_files = [1, 2, 3, 10, 16, 23, 30, 37, 128, 135, 142, 149, 156, 163,170, 177, 184,191, 198, 205 ] 


for file in csv_files:
    file = str(file)
    print("Removing cols for "+file+".csv")
    df = pd.read_csv("./hydrated/tweets-"+file+".csv", usecols = ['hashtags','place','text','user_description','user_location'])
    df.to_csv("./preprocessed_data/preproc_tweets-"+file+".csv", index=False)
