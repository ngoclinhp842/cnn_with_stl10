'''load_stl10_dataset.py
Downloads and extracts the STL-10 dataset (http://ai.stanford.edu/~acoates/stl10)
Oliver W. Layton
Original script from Martin Tutek (https://github.com/mttk/STL10)
'''

import sys
import os
import sys
import tarfile
import urllib.request as urllib

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

'''
Image constants
'''
HEIGHT = 96
WIDTH = 96
DEPTH = 3
SIZE = HEIGHT * WIDTH * DEPTH  # size of a single image in bytes


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    The image values are uint8s (0, 255)
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))

        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


def download_and_extract(DATA_URL, DATA_DIR, DATA_PATH, LABEL_PATH):
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename, float(count * block_size) / float(total_size)*100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)

    # Extract the tar file only if the extracted files do not already exist
    if not os.path.isfile(DATA_PATH) or not os.path.isfile(LABEL_PATH):
        print(f'Extracting {filepath}...', end='')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        print('Done!')


def save_images(images, labels):
    print("Saving images to disk...")

    # Make dataset/class/image subdirectories
    unique_labels = np.unique(labels)
    for label in unique_labels:
        directory = os.path.join('images', str(label))

        try:
            os.makedirs(directory, exist_ok=True)
        except:
            print(f'Error: Could not make label subdirectory {directory}! Exiting...')
            exit()

    for i, image in enumerate(images):
        filename = os.path.join('images', str(labels[i]), str(i) + '.png')

        currImg = Image.fromarray(image).save(filename)

        if i % 100 == 0:
            print(f'  Saved image {filename}')

    print("Done!")


def resize_images(imgs, scale_fact=3):
    ''' Rescales collection of images represented as a single ndarray

    Parameters:
    -----------
    imgs: ndarray. shape = (num images, x, y, color chan)
    scale_factor: downscale image resolution by this amount

    Returns:
    -----------
    scaled_imgs: ndarray. the downscaled images.
    '''
    if scale_fact == 1.0:
        print(f'preprocess_images: No resizing to do, scale factor = {scale_fact}.')
        return imgs

    print(f'Resizing {len(imgs)} images to {HEIGHT//scale_fact}x{WIDTH//scale_fact}...', end='')

    num_imgs = imgs.shape[0]
    scaled_imgs = np.zeros([num_imgs, HEIGHT//scale_fact, WIDTH//scale_fact, DEPTH], dtype=np.uint8)

    for i in range(num_imgs):
        currImg = Image.fromarray(imgs[i, :, :, :])
        currImg = currImg.resize(size=(HEIGHT//scale_fact, WIDTH//scale_fact))
        scaled_imgs[i, :, :, :] = np.array(currImg, dtype=np.uint8)

    print('Done!')
    return scaled_imgs


def purge_cached_dataset():
    CACHE_DIR = './numpy'
    img_cache_filename = os.path.join(CACHE_DIR, 'images.npy')
    label_cache_filename = os.path.join(CACHE_DIR, 'labels.npy')

    try:
        os.remove(img_cache_filename)
        os.remove(label_cache_filename)
    except OSError:
        pass


def load(save_imgs_to_disk=False, cache_binaries_to_disk=True, scale_fact=3):
    DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

    # local path to the directory with the data
    DATA_DIR = './data'

    CACHE_DIR = './numpy'
    img_cache_filename = os.path.join(CACHE_DIR, 'images.npy')
    label_cache_filename = os.path.join(CACHE_DIR, 'labels.npy')

    # local path to the binary train file with image data
    DATA_PATH = './data/stl10_binary/train_X.bin'

    # local path to the binary train file with labels
    LABEL_PATH = './data/stl10_binary/train_y.bin'

    # If we already saved the resized numpy arrays to disk, just load them and return
    if cache_binaries_to_disk and os.path.isfile(img_cache_filename) \
       and os.path.isfile(label_cache_filename):
        print(f'Found cached numpy arrays the images and labels. Loading them...')

        images = np.load(img_cache_filename)
        labels = np.load(label_cache_filename)

        print(f'Images are: {images.shape}')
        print(f'Labels are: {labels.shape}')

        return images, labels

    # download data if needed
    download_and_extract(DATA_URL, DATA_DIR, DATA_PATH, LABEL_PATH)

    # Read in the whole dataset and labels
    images = read_all_images(DATA_PATH)
    labels = read_labels(LABEL_PATH)

    print(f'Images are: {images.shape}')
    print(f'Labels are: {labels.shape}')

    # resize images to desired resolution, optionally save them to disk
    images = resize_images(images, scale_fact=scale_fact)

    # Save images to disk in PNG format
    if save_imgs_to_disk:
        save_images(images, labels)

    # Save the numpy array to disk in binary format to quickly reload the dataset next time
    if cache_binaries_to_disk:
        print(f'Saving Numpy arrays the images and labels to {CACHE_DIR}...', end='')
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
        except:
            print(f'Error: Could not make cache {CACHE_DIR}! Exiting...')
            exit()

        np.save(img_cache_filename, images)
        np.save(label_cache_filename, labels)
        print('Done!')

    return images, labels


if __name__ == "__main__":
    load(save_imgs_to_disk=True)
