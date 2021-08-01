from shutil import copyfile, move
import os
import random

#set directories
directory = str('/home/ubuntu/InstColorization/train_data/train2017')
target_directory_train = str('/home/ubuntu/InstColorization/train_data/train_small')
target_directory_val = str('/home/ubuntu/InstColorization/train_data/val_small')
target_directory_test = str('/home/ubuntu/InstColorization/test_small')
data_set_percent_size = float(0.05)

#print(os.listdir(directory))
# list all files in dir that are an image
files = [f for f in os.listdir(directory)]

#print(files)

# select a percent of the files randomly 
random_files = random.sample(files, int(len(files)*data_set_percent_size))
#random_files = np.random.choice(files, int(len(files)*data_set_percent_size))

#print(random_files)

# move the randomly selected images by renaming directory 

size = len(random_files)
train_size = int(size * 0.5)
val_size = int(size * 0.2)
test_size = int(size * 0.3)
train_files = random_files[:train_size]
val_files = random_files[train_size:train_size+val_size]
test_files = random_files[train_size+val_size:]

# print(train_files)
# print(val_files)
# print(test_files)

for file_name in train_files:      
    #print(directory+'/'+random_file_name)
    #print(target_directory+'/'+random_file_name)
    # os.rename(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    copyfile(directory+'/'+file_name, target_directory_train+'/'+file_name)
    # move(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    continue

for file_name in val_files:      
    #print(directory+'/'+random_file_name)
    #print(target_directory+'/'+random_file_name)
    # os.rename(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    copyfile(directory+'/'+file_name, target_directory_val+'/'+file_name)
    # move(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    continue

for file_name in test_files:      
    #print(directory+'/'+random_file_name)
    #print(target_directory+'/'+random_file_name)
    # os.rename(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    copyfile(directory+'/'+file_name, target_directory_test+'/'+file_name)
    # move(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    continue