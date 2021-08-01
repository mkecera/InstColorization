from shutil import copyfile, move
import os
import random

#set directories
directory = str('/home/ubuntu/InstColorization/train_data/train_tiny')
target_directory = str('/home/ubuntu/InstColorization/test_tiny')
data_set_percent_size = float(0.20)

#print(os.listdir(directory))
# list all files in dir that are an image
files = [f for f in os.listdir(directory)]

#print(files)

# select a percent of the files randomly 
random_files = random.sample(files, int(len(files)*data_set_percent_size))
#random_files = np.random.choice(files, int(len(files)*data_set_percent_size))

#print(random_files)

# move the randomly selected images by renaming directory 

for random_file_name in random_files:      
    #print(directory+'/'+random_file_name)
    #print(target_directory+'/'+random_file_name)
    # os.rename(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    # copyfile(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    move(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    continue