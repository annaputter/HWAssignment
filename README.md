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
As you can see in the code the network is evaluated by leaving a small part of the coordinate out of the training and 
measuring the loss function on them in the eval step.

#### b. representation projection 
In the paper the network is constructed from 4 layers with a sine activation function and one final linear layer that 
generates the output, hence we can say that:

I<sub>out</sub>[u,v] = &sum;<sub>n=0</sub><sup>256</sup>w<sub>n</sub>sin(f<sub>n</sub>(u,v))

please note that if we omit the 3 hidden layers then f<sub>n</sub> will be linear functions of u,v and the resulting 
representation will be an approximation of the fourier decomposition of the image.
With the hidden layers (actually it is true even with a single hidden layer) {f<sub>n</sub>(u,v)} are no longer linear, 
hence the resulting basis of the representation (sin(f<sub>n</sub>(u,v))) is not nesesesery orthogonal.

To represent all 100 images I divided the final layer of the network to 100 parallel brunches and the representation becomes:

I<sub>out</sub><sup>m</sup>[u,v] = &sum;<sub>n=0</sub><sup>256</sup>w<sup>m</sup><sub>n</sub>sin(f<sub>n</sub>(u,v))

In other words I am using a common basis {sin(f<sub>n</sub>(u,v))} for all the images, 
each image is represented by a set of coordinates in the space spun by the basis (the weights of the network). 

I ploted the weights in 3 different plots (one for each RGB channel), 
to view the plots set "experiment_num" in config.yaml to 0 and run test.py

### 2. Image interpolation
#### a. Image upsampling
To generate the upsampled images set "experiment_num" in config.yaml to 2, 
the images will be stored in 'experiments/exp_2' in your train_output_dir

#### b. Interpolation between pairs of images
The input of the network is only u,v the output are 100 RGB values (one for each image), choosing a single image I<sub>out</sub><sup>m</sup> is done by
multiplying the values by a vector of size 100X1 with 1 in the m-th value and 0 elsewhere.
Similarly, the interpolation between image m,k is done by multiplying the values by a vector of size 100X1 with &alpha; 
in the m-th value 1-&alpha; in the k-th value and 0 elsewhere. 
For the interpolation I choose the 3 pairs with the least euclidean distance between the weights of the uncommon layers.
To generate the interpolated images set "experiment_num" in config.yaml to 3,
the images will be stored in 'experiments/exp_3' in your train_output_dir

The results look good, however I believe that this due to the pair choice and not because of the interpolation itself (which is really naive). 

### 3. Improved image interpolation 
I used mse loss between the gt images and the reconstructed ones,
I think that if I would have used a combination of this loss with a loss on the derivatives of the image I would have gotten
an improvement in the interpolation (and probably in the upsampling as well).  