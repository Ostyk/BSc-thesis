# helper functions

import tensorflow as tf
import numpy as np
import os
import pandas as pd

def activate_layer(layer,image):
    '''
    Args:
    input image has to be of shape 227x277x3
    Returns:
    initialized activations from given layer
    '''
    return np.squeeze(sess.run(layer,feed_dict={x:image.reshape(1, 227,227,3)}))

def single_map(s_map, type_ = 'average'):
    '''
    Args: single layer map
    ie: (256,6,6)
    Returns: average individual maps, (256,) of 1 image
    '''
    if type_ == 'average':
        return np.array([np.average(s_map[:,:,i]) for i in range(s_map.shape[2])])
    elif type_ == 'maximum':
        return np.array([np.max(s_map[:,:,i]) for i in range(s_map.shape[2])])
    elif type_ == 'variance':
        return np.array([np.var(s_map[:,:,i]) for i in range(s_map.shape[2])])
    elif type_ == 'median':
        return np.array([np.median(s_map[:,:,i]) for i in range(s_map.shape[2])])

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def activate_layer(layer,image):
    '''
    Args:
    input image has to be of shape 227x277x3
    Returns:
    initialized activations from given layer
    '''
    return np.squeeze(sess.run(layer,feed_dict={x:image.reshape(1, 227,227,3)}))

def layer_params(layer,data_directory,output_size=4096,type_='average'):

    df = os.listdir(data_directory)
    data = [i for i in df if i[-3:] == 'jpg']
    q = int(len(df)-len(data))
    to_save_ks_test = np.zeros((len(list(data)),output_size))

    f = FloatProgress(min=0, max=len(df))
    display(f)
    im = 0
    for root, dirs, files in os.walk(data_directory):
        for index,i in enumerate(sorted(files)):
            path = root + "/" + i
            try:
                if type_ is None:
                    to_save_ks_test[index-q] = activate_layer(layer,imread(path))
                elif type_ == 'diff_pool':
                    r=activate_layer(layer,imread(path))
                    print(r.shape)
                    to_save_ks_test[index-q] = r.reshape([-1,])  #flatten
                else:
                    to_save_ks_test[index-q] = single_map(activate_layer(layer,imread(path)), type_)
                im+=1
            except ValueError:
                print("cannot identify {} as an image file".format(i))
            f.value+=1
    return to_save_ks_test, sorted([i[:-4] for i in data])
