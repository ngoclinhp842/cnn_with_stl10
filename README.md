# CNN with STL10

Built a modular convolutional neural network (CNN) for image classification using the STL-10 dataset, featuring convolution, max pooling, and gradient descent optimization.

## 🚀 Running CNN_with_STL10:
To run CNN_with_STL10 on the file main.py 

```sh
python3 main.py
```

## 👀 Example Running main.py:
<p align="center">
  <img align="center" alt="Accuracy over Epoch" width="400" src="https://github.com/ngoclinhp842/cnn_with_stl10/blob/main/outputs/running_example.png">
</p>

## 📝 Analysis:

1. Correlation between kernal size with accuracies 
The smaller kernel sizes seem to have better accuracies than the larger ones (3x3 and 5x5 vs 9x9 and 11x11).
Smaller kernels are more efficient at learning the finer, local features of the input data, leading to quicker convergence and better performance early on.
The larger kernels also have slower improement and more fluctations which may imply they are more prone to overfitting.

<p align="center">
  <img align="center" alt="Kernal Size" width="400" src="https://github.com/ngoclinhp842/cnn_with_stl10/blob/main/outputs/train_accuracy_across_different_number_of_kernals.png">
</p>

2. Visualization of kernals

<p align="center">
  <img align="center" alt="Kernal Visualization" width="400" src="https://github.com/ngoclinhp842/cnn_with_stl10/blob/main/outputs/kernals.png">
</p>

## ⚖️ License:
Apache License 2.0
