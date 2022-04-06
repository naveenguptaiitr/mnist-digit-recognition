# Python utility file for supporting functions or method

import numpy as np

# One Hot Encoding Method 
def oneHotEncoder(num_list):
    
    shape = (num_list.size, 10)
    num_rows = np.arange(num_list.size)
    train_target_1hot = np.zeros(shape)
    train_target_1hot[num_rows, num_list] = 1
    
    return train_target_1hot
    
#Unwrapping downloaded data     
def unwrapping_data(input_data, target_data, sampling_fr):
    
    index_sampler_set = np.arange(input_data.shape[0])
    index_sampler = np.random.choice(index_sampler_set, int(input_data.shape[0]*sampling_fr), replace=False)
    
    input_data = [*input_data[index_sampler]]
    output_data = [*target_data[index_sampler]]
    
    input_data_np = np.array(input_data)
    output_data_np = np.array(output_data)
    
    output_data_one_hot_np = oneHotEncoder(output_data_np)
    
    return input_data_np, output_data_one_hot_np


#Data shuffling function
def data_shuffling(input_data, label_data):
    
    random_index = np.random.choice(np.arange(input_data.shape[0]), input_data.shape[0], replace=False)
    input_shuffled = np.array([*input_data[random_index]])
    label_shuffled = np.array([*label_data[random_index]])
        
    return input_shuffled, label_shuffled
    