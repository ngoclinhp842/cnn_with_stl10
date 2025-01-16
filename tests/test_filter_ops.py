'''
test_filter_ops.py

This script contains unit tests for the filter_ops.py module. It is designed to ensure
that the functions and logic within filter_ops.py behave as expected.

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
from filter_ops import conv2_gray, conv2, conv2nn, max_pool, max_poolnn

plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])
plt.rcParams.update({'font.size': 20})

np.set_printoptions(suppress=True, precision=3)

class TestConv(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        # generates 4 (total) Gabor filters that respond to
        self.gabor_kers = np.zeros((4, 21, 21))
        # horizontal
        self.gabor_kers[0, :, :] = self.gabor(filter_sz_xy=[21,21], w=2, theta=-np.pi/2)
        # -45° (negative slope)
        self.gabor_kers[1, :, :] = self.gabor(filter_sz_xy=[21,21], w=2, theta=-np.pi / 4)
        # vertical
        self.gabor_kers[2, :, :] = self.gabor(filter_sz_xy=[21,21], w=2, theta=0)
        # 45° (positive slope)
        self.gabor_kers[3, :, :] = self.gabor(filter_sz_xy=[21,21], w=2, theta=-np.pi  * 3/4)

        sobel_ker_1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_ker_2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_ker_3 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
        sobel_ker_4 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
        self.sobel_ops = np.stack((sobel_ker_1, sobel_ker_2, sobel_ker_3, sobel_ker_4), axis=0)

        # Make a new axis for color channel and replicate the Sobel filters across that channel so that each RGB channel
        # is filtered by the same filters
        self.sobel_ops_chans = np.tile(self.sobel_ops[:, np.newaxis, :, :], (3, 1, 1, ))

        clownfish = Image.open('images/clownfish.png', 'r')
        clownfish_np = clownfish.convert('RGB')
        plt.imshow(clownfish_np)
        self.clownfish_np = np.transpose(clownfish_np, (2, 0, 1))

    def gabor(filter_sz_xy, w, theta, K=np.pi):
        ''' Generates a simple cell-like Gabor filter.

        Parameters:
        -----------
        filter_sz_xy: tuple. shape=(filter width, filter height)
        w: float. spatial frequency of the filter
        theta: float. Angular direction of the filter in radians
        K: float. Angular phase of the filter in radians
        '''
        rad_x, rad_y = filter_sz_xy
        rad_x, rad_y = int(rad_x/2), int(rad_y/2)
        [x, y] = np.meshgrid(np.arange(-rad_x, rad_x+1), np.arange(-rad_y, rad_y+1))

        x_p = x*np.cos(theta) + y*np.sin(theta)
        y_p = -x*np.sin(theta) + y*np.cos(theta)

        # Take the real part of the filter
        gauss = w**2 / (4*np.pi*K**2) * np.exp(-(w**2/(8*K**2)) * (4*x_p**2 + y_p**2))
        sinusoid = np.cos(w*x_p) * np.exp(K**2/2)
        gabor = gauss * sinusoid
        return gabor

    def plot_image_panel(imgs):
        '''Plot a few images side by side horizontally'''
        fig, axes = plt.subplots(ncols=len(imgs), figsize=(3*len(imgs),6))
        if len(imgs) == 1:
            axes.imshow(imgs[0])
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            for ax, img in zip(axes, imgs):
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
        plt.show()

    def test_single_2d_conv_odd_size(self):
        """
        Test filter_ops.py conv2_gray with a single uniform normalized averaging kernel and an odd-sized image.

        This test works as follows:
        1. Create a test image with a 'staircase' pattern of numbers from 0 to 10 across the column
           dimension and replicate it across rows.
        2. Create a single uniform (normalized) averaging kernel.
        3. Print the shape of the test image and kernel.
        4. Print the test image and kernel.
        5. Verify the output of the convolution is the same as scipy.signal.convolve2d.

        Verifies that the output image of the convolution has the same shape as the input image.
        """
        # Create test 'staircase' pattern 0-10 across the column dimension and replicate across rows
        test_num_cols = 7
        test_img = np.tile(1 + np.arange(0, test_num_cols), (test_num_cols, 1))

        # Test single kernal, odd size
        # Single uniform (normalized) averaging kernel
        test_ker_sz = 2
        test_ker = np.ones([1, test_ker_sz, test_ker_sz])
        # Normalize the kernel
        test_ker /= test_ker.sum()
        self.assertEqual(test_img.shape, test_img)
        self.assertEqual(test_ker.shape, test_ker)

        # Convolve the test image with the kernel using conv2_gray
        test_out_img = conv2_gray(test_img, test_ker, verbose=True)

        # Verify the output image has the same shape as the input image
        self.assertEqual(test_out_img.shape, (1, 7, 7))

        # Verify the output of the convolution is the same as scipy.signal.convolve2d
        self.assertEqual(np.allclose(test_out_img[0]), sp.convolve2d(test_img, test_ker[0], mode='same'))

    def test_single_2d_conv_even_size(self):
        """
        Test filter_ops.py conv2_gray with a single uniform normalized averaging kernel and an even-sized image.

        This test works as follows:
        1. Create a test image with a 'staircase' pattern of numbers from 0 to 10 across the column
           dimension and replicate it across rows.
        2. Create a single uniform (normalized) averaging kernel.
        3. Print the shape of the test image and kernel.
        4. Print the test image and kernel.
        5. Verify the output of the convolution is the same as scipy.signal.convolve2d.
        6. Verify that the output image of the convolution has the same shape as the input image.
        """

        # Create test 'staircase' pattern 0-10 across the column dimension and replicate across rows
        test_num_cols = 8
        test_img = np.tile(1 + np.arange(0, test_num_cols), (test_num_cols, 1))
        print(f'test img shape is {test_img.shape}')
        print('test img looks like:')
        print(test_img)

        # Test single kernel, even size
        # Single uniform (normalized) averaging kernel
        test_ker_sz = 2
        test_ker = np.ones([1, test_ker_sz, test_ker_sz])

        # Normalize the kernel
        test_ker /= test_ker.sum()
        print(f'test ker shape is {test_ker.shape}')
        print('test ker looks like:')
        print(test_ker)

        # Convolve the test image with the kernel using conv2_gray
        test_out_img = conv2_gray(test_img, test_ker, verbose=True)

        # Verify the output image has the same shape as the input image
        self.assertEqual(test_out_img.shape, (1, 8, 8))

        # Verify the output of the convolution is the same as scipy.signal.convolve2d
        self.assertEqual(np.allclose(test_out_img[0], sp.convolve2d(test_img, test_ker[0], mode='same')), True)

    def test_odd_single_kernal_with_even_image_size(self):
        """
        Test filter_ops.py conv2_gray with a single uniform normalized averaging kernel (odd size) and an even-sized image.

        This test works as follows:
        1. Create a test image with a 'staircase' pattern of numbers from 0 to 10 across the column
           dimension and replicate it across rows.
        2. Create a single uniform (normalized) averaging kernel.
        3. Print the shape of the test image and kernel.
        4. Print the test image and kernel.
        5. Verify the output of the convolution is the same as scipy.signal.convolve2d.
        6. Verify that the output image of the convolution. 
        """
        # Create test 'staircase' pattern 0-10 across the column dimension and replicate across rows
        test_num_cols = 8
        test_img = np.tile(1 + np.arange(0, test_num_cols), (test_num_cols+2, 1))

        test_ker2_sz = 5
        test_ker2 = np.ones([1, test_ker2_sz, test_ker2_sz])
        test_ker2 /= test_ker2.sum()
        print(f'test img shape is {test_img.shape} and test ker shape is {test_ker2.shape}')
        print(f'test img looks like:\n{test_img}\nand test ker looks like\n{test_ker2}')

        # Test single kernel odd size, even image size
        test_out_img = conv2_gray(test_img, test_ker2, verbose=True)

        # Verify the output image 
        self.assertEqual(test_out_img.shape, (1, 10, 8))

        # Verify the output of the convolution is the same as scipy.signal.convolve2d
        self.assertEqual(np.allclose(test_out_img[0], sp.convolve2d(test_img, test_ker2[0], mode='same')), True)

    def test_multiple_kernels_grey_scale(self):
        """
        Test filter_ops.py conv2_gray with multiple kernels and a grayscale image.

        This test works as follows:
        1. Load in the clownfish image and convert it to grayscale.
        2. Create a list of multiple Gabor filters that respond to different orientations.
        3. Convolve the grayscale image with the filters using conv2_gray.
        4. Verify that the output of the convolution is the same as scipy.signal.convolve2d.
        5. Verify that the output image of the convolution.
        """
        # Load in the clownfish image then convert to grayscale for testing
        clownfish = Image.open('images/clownfish.png', 'r')
        clownfish_gray = clownfish.convert('L')  # convert to grayscale
        # show clownfish_gray
        clownfish_gray.show()

        # Create a list of multiple Gabor filters that respond to different orientations
        gabor_kers = []
        for th in np.arange(-np.pi/2, np.pi/2, np.pi/8):
            print(th)
            gabor_kers.append(self.gabor(filter_sz_xy=[121, 121], w=0.25, theta=th))
        self.plot_image_panel(gabor_kers)

        self.plot_image_panel(self.gabor_kers)
        
        gabor_expected_kers = np.array([
            [-1.524,  4.484, -1.524],
            [ 0.616,  4.484,  0.616],
            [ 4.263,  4.484,  4.263],
            [ 0.616,  4.484,  0.616]
        ])

        self.assertEqual(gabor_kers.shape, (4, 21, 21))
        self.assertEqual(gabor_kers[:,9:12, 10], gabor_expected_kers)

    def test_multiple_kernels_color(self):
        """
        Test filter_ops.py conv2 with multiple kernels and a color image.

        This test works as follows:
        1. Load in the clownfish image and convert it to RGB.
        2. Create a list of multiple Gabor filters that respond to different orientations.
        3. Convolve the RGB image with the filters using conv2.
        4. Verify that the output of the convolution is the same as scipy.signal.convolve2d.
        5. Verify that the output image of the convolution has the same shape as the input image.
        """
        # drop the 4th color channel (alpha channel) so that ony RGB channels remain
        
        clownfish = Image.open('images/clownfish.png', 'r')
        clownfish_np = clownfish.convert('RGB')
        plt.imshow(clownfish_np)
        clownfish_np = np.transpose(clownfish_np, (2, 0, 1))  # permute dims so that channel is leading
        self.assertEqual(clownfish_np.shape, (3, 238, 241))

        # Box filter test
        # Make a 11x11 box filter, with constant, identical positive values 
        # normalized so that the entire filter sums to 1.
        # Add a leading singleton dimension so shape is (1, 11, 11)
        box_ker = np.full((1, 11, 11), 1 / 121)

        self.assertEqual(box_ker.shape, (1, 11, 11))
        print(f'box ker looks like\n{box_ker}')     

        clownfish_out = conv2(clownfish_np, box_ker)
        self.assertEqual(clownfish_out.shape, (1, 3, 238, 241))

        # visualize output of RGB box filter convolution of clownfish image
        clownfish_out = clownfish_out.astype(np.uint8)
        clownfish_out_reordered = np.transpose(clownfish_out, (0, 2, 3, 1))
        print('After reordering, clownfish_out shape is', clownfish_out_reordered.shape)
        self.plot_image_panel(clownfish_out_reordered)

        clownfish_gabor_out = conv2(clownfish_np, self.gabor_kers)
        self.assertEqual(clownfish_gabor_out.shape, (4, 3, 238, 241))

        # visualize output RGB Gabor filter convolution of clownfish image
        # Compute the minimum and maximum values across the RGB channels for each pixel
        min_vals = np.min(clownfish_gabor_out, axis=-1, keepdims=True)
        max_vals = np.max(clownfish_gabor_out, axis=-1, keepdims=True)
        # Apply min-max normalization to scale values to [0, 1] within each RGB triplet
        normalized_output = (clownfish_gabor_out - min_vals) / (max_vals - min_vals + 1e-8)
        # Scale to the range [0, 255] and convert to uint8
        normalized_output = (normalized_output * 255)
        # Re-order the dimensions for visualization so that they are: n_kers, img_y, img_x, n_chans (4, 238, 241, 4)
        clownfish_gabor_out_reordered = np.transpose(clownfish_gabor_out, (0, 2, 3, 1))
        # converted to unint8
        clownfish_gabor_out_reordered = clownfish_gabor_out_reordered.astype(np.uint8)
        self.plot_image_panel(clownfish_gabor_out_reordered)

    def test_sobel_filters(self):
        """
        Test the Sobel filters are correctly constructed.

        This test works as follows:
        1. Print out the shape of the Sobel filters.
        2. Visualize the Sobel filters.
        """
        print(f'Test Sobel filters shape (K, D, k_y, k_x): {self.sobel_ops_chans.shape}')

        self.plot_image_panel(self.sobel_ops)

    def test_multiple_images_sobel_filters(self):
        """
        Test that the Sobel filters with color channels works on multiple images

        This test works as follows:
        1. Stack two copies of the clownfish image to test on multiple images.
        2. Convolve the images with the Sobel filters with color channels using conv2nn.
        3. Verify that the output of the convolution is the same for the two input images.
        4. Verify that the output image of the convolution has the same shape as the input image.
        """
        imgs = np.stack([self.clownfish_np, self.clownfish_np])
        bias = np.zeros(len(self.gabor_kers))
        print(f'Test img shape is {imgs.shape}')

        clownfish_color_imgs_out = conv2nn(imgs, self.sobel_ops_chans, bias)
        self.assertEqual(clownfish_color_imgs_out.shape, (2, 4, 238, 241))
        if np.all(clownfish_color_imgs_out[0] == clownfish_color_imgs_out[1]):
            print('Your filter maps are identical as expected!')
        else:
            print('Your filter maps are not the same :(')

        # test values in the 10th row of the first filter output
        firstKerOutTest = clownfish_color_imgs_out[0, 0, 9, :10]
        print(f'First few values in 10th row of your 1st filter output are:\n{firstKerOutTest}')
        self.assertEqual(firstKerOutTest, np.ndarray([-660., 0., -16., 4., 4., 1., -7., -1., -4., -10.]))

        # test values in the 5th row of 2nd filter output
        secondKerOutTest = clownfish_color_imgs_out[0, 0, 4, :10]
        print(f'First few values in 5th row of 2nd filter output:\n{secondKerOutTest}')
        self.assertEqual(secondKerOutTest, np.ndarray([-814., -60., 12., 63., 63., 34., -126., -204., 49., 196.]))

        # visualize the output of the convolution
        self.plot_image_panel(clownfish_color_imgs_out[0])

    def test_max_pool(self):
        """
        Test the max_pool function with various patterns and stride values.

        This test verifies:
        - The shape of the output after pooling.
        - The correctness of the pooled values.
        """

        # Create simple checkerboard pattern for testing.
        # NOTE: We're creating an extra singleton dimension 
        grid = np.tile(np.stack([np.array([1, 0] * 5), np.array([0, 0] * 5)]), (3, 1))
        print(f'Image shape is {grid.shape}')
        print(f'The checkerboard image looks like\n{grid}')

        # Test max pooling with default stride
        grid_pooled = max_pool(grid)
        self.assertEqual(grid_pooled.shape, (5, 9))
        self.assertEqual(grid_pooled, np.ones((5, 9)))

        # Test max pooling with stride of 2
        grid_pooled = max_pool(grid, strides=2)
        self.assertEqual(grid_pooled.shape, (3, 5))
        self.assertEqual(grid_pooled, np.ones((3, 5)))

        # Create simple odd checkerboard pattern for testing
        grid_odd = np.tile(np.stack([np.array([1, 0] * 2), np.array([0, 0] * 2), np.array([1, 0] * 2)]), (3, 1))
        print(f'Image shape is {grid_odd.shape}')
        print(f'The odd checkerboard image looks like\n{grid_odd}')

        # Test max pooling with default stride on the odd pattern
        grid_pooled = max_pool(grid_odd)
        self.assertEqual(grid_pooled.shape, (8, 3))
        self.assertEqual(grid_pooled, np.ones((8, 3)))

        # Test max pooling with stride of 2 on the odd pattern
        grid_pooled = max_pool(grid_odd, strides=2)
        self.assertEqual(grid_pooled.shape, (4, 2))
        self.assertEqual(grid_pooled, np.ones((4, 2)))

        # Create simple bar pattern for testing
        bars = np.tile(np.stack([np.array([1] * 10), np.array([0] * 10), np.array([0] * 10), np.array([0] * 10)]), (3, 1))
        print(f'Image shape is {bars.shape}')
        print(f'Here are your bar inputs:\n{bars}')

        # Test max pooling with default stride on bar pattern
        bars_pooled = max_pool(bars)
        self.assertEqual(bars_pooled.shape, (11, 9))
        self.assertEqual(bars_pooled, np.ones((11, 9)))

        # Test max pooling with stride of 2 on bar pattern
        bars_pooled = max_pool(bars, strides=2)
        self.assertEqual(bars_pooled.shape, (6, 5))
        print(f'Here are your max filtered bars for testing strides of 2:\n{bars_pooled}')

        # Test max pooling with stride of 3 on bar pattern
        bars_pooled = max_pool(bars, strides=3)
        self.assertEqual(bars_pooled.shape, (4, 3))
        print(f'Here are your max filtered bars for testing strides of 3:\n{bars_pooled}')


    def test_max_poolnn(self):
        """
        Test the max_poolnn function with various patterns and stride values.

        This test verifies:
        - The shape of the output after pooling.
        - The correctness of the pooled values.
        """
        # Create simple bar pattern for testing
        bars_batch = np.tile(np.stack([np.array([1]*10), np.array([0]*10), np.array([0]*10), np.array([0]*10)]), (1, 1, 3, 1))
        print(f'Image shape is {bars_batch.shape}')
        print(f'Here are your bar inputs:\n{bars_batch[0,0]}')

        bars_batch_pooled = max_poolnn(bars_batch, strides=2, pool_size=2, verbose=False)
        self.assertEqual(bars_batch_pooled.shape, (1, 1, 6, 5))
        print(f'Here are your max filtered bars:\n{bars_batch_pooled[0,0]}')

        # Test on real image
        clownfish = Image.open('images/clownfish.png', 'r')
        clownfish_np_batch = np.array(clownfish)[:,:,:3]
        clownfish_np_batch = np.moveaxis(clownfish_np_batch,2,0)
        clownfish_np_batch = np.expand_dims(np.array(clownfish_np_batch), 0)
        print(f'Shape of clownfish image is {clownfish_np_batch.shape}')

        clownfish_mp = max_poolnn(clownfish_np_batch, pool_size=8)
        self.assertEqual(clownfish_mp.shape, (1, 3, 231, 234))
        self.assertEqual(clownfish_mp[0,1,9,:10], np.ndarray([170., 170., 170., 170., 170., 170., 118.,  58.,  54.,  54.]))
        self.assertEqual(clownfish_mp[0,0,9,:10], np.ndarray([105., 105., 105., 105., 105., 105.,  80.,  28.,  22.,  22.]))


if __name__ == '__main__':
    unittest.main()