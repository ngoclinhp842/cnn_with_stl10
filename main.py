'''
main.py

This script serves as the entry point for the program. It contains the main() function,
which is executed when the script is run directly. Additional logic and functionality
can be added within the main() function or through imported modules.

Author: Michelle Phan
Date: Fall 2024
Version: 1.0
'''
import numpy as np
import matplotlib.pyplot as plt

plt.show()
plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])
plt.rcParams.update({'font.size': 20})

import load_stl10_dataset
import preprocess_data
from network import ConvNet4AccelV2

def plot_weights(wts, saveFig=True, filename='convWts.png'):
    """
    Plot the weights as images in a grid format.

    Parameters:
    -----------
    wts : list or ndarray
        List or array of weights to plot. Each weight is expected to be an image.
    saveFig : bool, optional
        Whether to save the figure to a file. Default is True.
    filename : str, optional
        The filename to save the figure if saveFig is True. Default is 'convWts.png'.

    Returns:
    --------
    None
    """
    # Determine the grid size
    grid_sz = int(np.sqrt(len(wts)))
    
    # Create a new figure
    plt.figure(figsize=(5, 5))
    
    # Iterate over the grid and plot each weight
    for x in range(grid_sz):
        for y in range(grid_sz):
            # Compute linear index for the current subplot
            lin_ind = np.ravel_multi_index((x, y), dims=(grid_sz, grid_sz))
            plt.subplot(grid_sz, grid_sz, lin_ind + 1)
            
            # Process and normalize the current weight image
            currImg = wts[lin_ind]
            low, high = np.min(currImg), np.max(currImg)
            currImg = 255 * (currImg - low) / (high - low)
            currImg = currImg.astype('uint8')
            
            # Display the image
            plt.imshow(currImg)
            plt.gca().axis('off')  # Hide axis
    
    # Adjust layout spacing
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    
    # Save the figure if requested
    if saveFig:
        plt.savefig(filename)
    
    # Show the plot
    plt.show()

def main():
    """
    Main function for the program.

    This function contains the main logic of the program. It loads the dataset,
    preprocesses the data, creates and trains a ConvNet4AccelV2 object, and
    visualizes the weights and accuracy of the network.
    """
    # Load the dataset
    x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev = preprocess_data.load_stl10(
        n_train_samps=4578, n_test_samps=400, n_valid_samps=2, n_dev_samps=20, scale_fact=6)

    # Create and train a ConvNet4AccelV2 object for each kernel size
    for kernel_size in [3, 5, 7, 9, 11]:
        # Create a ConvNet4AccelV2 object
        net = ConvNet4AccelV2(input_shape=(3, 32, 32), n_kers=(kernel_size**2,), n_classes=10,
                              verbose=False, dropout_rate=0.0, r_seed=0)

        # Compile the network
        net.compile(optimizer_name='adam', lr=1e-2)

        # Train the network
        loss_history, train_acc_history, val_acc = net.fit(x_dev, y_dev, x_val, y_val,
                                                          mini_batch_sz=15, n_epochs=20, acc_freq=2, print_every=6)

        # Plot the weights of the network
        plot_weights(net.layers[0].wts.transpose(0, 2, 3, 1), saveFig=True,
                     filename=f'filters_kernelsize_{kernel_size}.png')

    # Plot the train accuracy for different kernel sizes
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot the validation accuracy for each network
    for kernel_size in [3, 5, 7, 9, 11]:
        # Load the training accuracy history for the current kernel size
        train_acc_history = np.load(f'train_accuracy_kernelsize_{kernel_size}.npy')

        # Plot the training accuracy
        plt.plot(train_acc_history, label=f'Kernel Size {kernel_size}x{kernel_size}',
                 linestyle='-', marker='o')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy for Different Kernel Sizes')
    plt.legend(loc='lower right')

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()