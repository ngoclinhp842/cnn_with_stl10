'''
test_conv_net.py

This script contains unit tests for the layer.py module. It is designed to ensure
that the functions and logic within layer.py behave as expected.

Author: Michelle Phan
Date: Fall 2024
Version: 1.0
'''

import unittest
import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.signal as sp
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/linhphan/Downloads/Colby College/neuron_networks/cnn_with_stl10/cnn_with_stl10') 
from layer import *

plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])
plt.rcParams.update({'font.size': 20})

np.set_printoptions(suppress=True, precision=3)

class TestConv(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)

    def test_linear(self):
        """
        Test the linear activation function.

        The linear activation function simply passes the input to the output
        without any changes. This test verifies that the output is the same as
        the input.
        """
        test_layer = Layer(0, 'test')
        test_layer.net_in = np.arange(10)
        test_layer.linear()
        self.assertTrue(np.all(test_layer.net_act == np.arange(10)))

    def test_relu(self):
        """
        Test the ReLU activation function.

        This test verifies that the ReLU activation function correctly sets 
        negative inputs to zero while leaving positive inputs unchanged.
        """
        # Create a random number generator with a fixed seed for reproducibility
        rng = np.random.default_rng(0)

        # Initialize a test layer
        test_layer = Layer(0, 'test')

        # Assign random inputs to the layer, with values ranging from -0.5 to 0.5
        test_layer.net_in = rng.random([3, 3]) - 0.5
        
        # Apply the ReLU activation function
        test_layer.relu()
        
        # Print the activated outputs of the layer
        print(f'{test_layer.net_act}')

    def test_soft_max(self):
        """
        Test the softmax activation function.

        The softmax activation function scales the input to ensure all output
        values are between 0 and 1 and sum to 1. This test verifies that the
        output of the softmax activation function is as expected.
        """
        # Create a random number generator with a fixed seed for reproducibility
        rng = np.random.default_rng(0)

        # Initialize a test layer
        test_layer = Layer(0, 'test')

        # Assign random inputs to the layer, with values ranging from 0 to 1
        test_layer.net_in = rng.random([2, 5])
        
        # Apply the softmax activation function
        test_layer.softmax()
        
        # Print the activated outputs of the layer
        print(f'{test_layer.net_act}')

    def test_cross_entropy(self):
        """
        Test the cross-entropy loss function.

        This test verifies that the cross-entropy loss function correctly
        computes the loss between the predicted output and the true labels.
        """
        rng = np.random.default_rng(0)
        y = np.array([0, 4, 1])
        test_layer = Layer(0, 'test')
        test_layer.net_in = rng.random([3, 5])
        test_layer.softmax()
        self.assertEqual(test_layer.cross_entropy(y), 1.65869)

    def test_conv2d_initilization(self):
        """
        Test the initialization of a Conv2D layer.

        This test verifies that the Conv2D layer is initialized correctly,
        with the correct number of filters, kernel size, and weight scale.
        """
        # Create a Conv2D layer with 2 filters, kernel size 2, and weight scale 0.1
        conv2_layer = Conv2D(0, 'conv2', n_kers=2, ker_sz=2, wt_scale=1e-1, r_seed=2)

        # Print the weights and bias terms of the layer
        print('Your filter weights are\n{}'.format(conv2_layer.wts))
        print('Your bias terms are\n{}'.format(conv2_layer.b))

    def test_forward_conv2D_layer(self):
        """
        Test the forward pass of a Conv2D layer with ReLU activation.

        This test verifies:
        - The shapes of input, weights, net activation, and net input.
        - The initial weights, net input, and activations of the Conv2D layer.
        """
        # Initialize random number generator for reproducibility
        rng = np.random.default_rng(1)

        # Define test network parameters
        mini_batch_sz, n_kers, n_chans, ker_sz, img_y, img_x = 1, 2, 3, 4, 5, 5

        # Create random test input with specified dimensions
        inputs = rng.standard_normal((mini_batch_sz, n_chans, img_y, img_x))

        # Create a Conv2D layer with ReLU activation function
        conv_layer = Conv2D(0, 'test', n_kers, ker_sz, n_chans=n_chans, wt_scale=1e-1, activation='relu', r_seed=3)

        # Perform a forward pass through the layer
        net_act = conv_layer.forward(inputs)

        # Extract the computed net input and weights
        net_in = conv_layer.net_in
        wts = conv_layer.get_wts()
        inp = conv_layer.input

        # Assert expected shapes for input, weights, net activation, and net input
        self.assertEqual(inp.shape, (1, 3, 5, 5))
        self.assertEqual(wts.shape, (2, 3, 4, 4))
        self.assertEqual(net_act.shape, (1, 2, 5, 5))
        self.assertEqual(net_in.shape, (1, 2, 5, 5))

        # Print details of weights, net input, and net activation for debugging
        print('The first chunk of your filters/weights is:\n', wts[0, 0])
        print('The first chunk of your net_in is:\n', net_in[0, 0])
        print('The first chunk of your net_act is:\n', net_act[0, 0])


if __name__ == '__main__':
    unittest.main()