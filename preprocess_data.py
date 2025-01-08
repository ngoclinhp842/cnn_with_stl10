'''preprocess_data.py
Preprocessing data in STL-10 image dataset
Michelle Phan and Varsha Yarram 
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np
import load_stl10_dataset


def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N, RGB color chans, height, width)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9

    TODO:
    1) Cast imgs to float64
    2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    3) Treating the pixels as features, standardize the features "seperately"
    4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1,2,...10
    '''
    # 1) Cast imgs to float64
    imgs = imgs.astype(np.float64)
    
    # 2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    num_imgs, height, width, chans = imgs.shape
    imgs = imgs.reshape(num_imgs, chans, height, width)
    
    # 3) Treating the pixels as features, standardize the features separately
    imgs = (imgs - np.mean(imgs, axis=0)) / np.std(imgs, axis = 0)

    # 4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1, 2, ..., 10
    # Assuming the input labels are 1-indexed (1, 2, ..., 10)
    labels = labels - 1  # Shift to 0-indexed (0, 1, ..., 9)
    
    return imgs, labels


def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)

    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)

    TODO:
    1) Divvy up the images into train/test/validation/development non-overlapping subsets (see return vars)

    NOTE: Resist the urge to shuffle the data here! It is best to re-shuffle the data after
    each epoch of training so hold off on shuffling until you train your neural network.
    '''

    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return None
  
    x_train = data[:n_train_samps]
    y_train = y[:n_train_samps]

    x_test = data[n_train_samps:n_train_samps + n_test_samps]
    y_test = y[n_train_samps:n_train_samps + n_test_samps]

    x_val = data[n_train_samps + n_test_samps:n_train_samps + n_test_samps + n_valid_samps]
    y_val = y[n_train_samps + n_test_samps:n_train_samps + n_test_samps + n_valid_samps]

    x_dev = data[n_train_samps + n_test_samps + n_valid_samps:]
    y_dev = y[n_train_samps + n_test_samps + n_valid_samps:]

    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev


def load_stl10(n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500, scale_fact=3):
    '''Automates the process of:
    - loading in the STL-10 dataset and labels
    - preprocessing
    - creating the train/test/validation/dev splits.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''

    #loading in STL-10 dataset and labels
    imgs, labels = load_stl10_dataset.load(scale_fact=scale_fact)
    
    #preprocessing
    imgs, labels = preprocess_stl(imgs, labels)

    #creating the train/test/validation/dev splits
    result = create_splits(imgs, labels, n_train_samps, n_test_samps, n_valid_samps, n_dev_samps)

    if result is None:
        print("Error creating dataset splits!")
        return None

    return result
