'''
Given a mobilenet formatted numpy image, 
perform some basic analysis
'''

import numpy as np
from scipy.stats import skew, kurtosis

def basic_image_features(image_batch):
    '''
    Return the image-wise mean, variance, skewness and kurtosis
    for each of RGB
    INPUTS:
    image_batch : Numpy Array.
                    Follows the mobilenet format:
                    (num_frames, target_size[0], target_size[1], 3)
    
    OUTPUTS:
    features    : Numpy Array.
                    (num_frames, num_features)
    '''
    means = np.mean(image_batch, axis=(1,2))
    variances = np.var(image_batch, axis=(1,2))
    # kurtosis, skewness do not support multiple axes
    # need to reshape them to (num_frames, 224*224, 3)
    tmp_batch = np.reshape(image_batch, (image_batch.shape[0],
                                         -1, image_batch.shape[3]))
    kurtosises = kurtosis(tmp_batch, axis=1)
    skewnesses = skew(tmp_batch, axis=1)
    # stack horizontally and return answer
    return np.hstack([means, variances, kurtosises, skewnesses])