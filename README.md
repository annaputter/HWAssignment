# HW assignment - Anna Cohen
## How to run the Code
All the configurations for the code are in config.yaml
To run the code please set the parameters:
dataset_dir - to the location of the data
train_output_dir - to directory where outputs (such as weights and outputs for the different questions) will be stored

1. To train the network run train.py
2. To generate the reconstructed images set "experiment_num" in config.yaml to 1 and run test.py, 
   the images will be stored in 'experiments/exp_1' in your train_output_dir 

## Questions
### 1. Image representation
#### a. generalization
As you can see in the code evaluated the network by measuring the maximus mse among the 100 images, 
it was very easy to implement but it is not the best way,
I think that a better measure would have been to compare the weights of the final layers to the fourier decomposition of each image.

#### b. representation projection 
As you can see in the code difference between the images is the final layer of the network, and since the last common
layer ends with a sine activation it is equivalent to say that the difference of the representations is the weights of 
256 sinuses that represent each coordinate set. Although I didn't do it, it is possible to use those weights and the 
outputs of the last common layer to estimate the fourier decomposition of each image.
I ploted the weights in 3 different plots (one for each RGB channel), 
to view the plots set "experiment_num" in config.yaml to 0 and run test.py

### 2. Image interpolation
#### a. Image upsampling
To generate the upsampled images set "experiment_num" in config.yaml to 2, 
the images will be stored in 'experiments/exp_2' in your train_output_dir

#### b. Interpolation between pairs of images
For the interpolation I choose the 3 pairs with the least euclidean distance between the weights of the uncommon layers.
To generate the interpolated images set "experiment_num" in config.yaml to 3,
the images will be stored in 'experiments/exp_3' in your train_output_dir

### 3. Improved image interpolation 
first of all I used mse loss between the gt images and the reconstructed ones,
I think that if I would have used a combination of this loss with a loss on the derivatives of the image I would have gotten
an improvment in the interpolation (and probably in the upsampling as well). 
 