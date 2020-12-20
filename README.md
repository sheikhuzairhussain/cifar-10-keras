# Image classifier with Keras and CIFAR-10 dataset

This is an **image classification program** built with a  **Convolutional Neural Network (CNN)**. It uses [TensorFlow](https://www.tensorflow.org/) (with [Keras](https://keras.io/)) to train a CNN with images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and make predictions accordingly. The images are classified into 10 classes - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

<p align="center">
<br>
<img src="https://i.ibb.co/72H1FSy/frog.png" width="70%">
</p>

**Requirements:** Python 3 with TensorFlow and Matplotlib installed.
## Usage
Clone the repo and enter the directory:
```
git clone https://github.com/sheikhuzairhussain/cifar-10-keras.git
cd cifar-10-keras
```
To classify an image:
```
python predict.py <path_to_your_image>
```
To train the model yourself (optional):
```
python train.py
```
**Note:** To test the dataset, run the `test()` function in `train.py`

