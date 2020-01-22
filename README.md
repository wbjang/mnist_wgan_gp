### Wasserstein GAN - Gradient Penalty (WGAN-GP) for MNIST

Based on this Paper : https://arxiv.org/pdf/1704.00028.pdf

#### Brief Introduction

The vanilla GAN (https://arxiv.org/abs/1406.2661) tries to find the Nash Equilibrium between Generator and Discriminator, and it minimizes the Jessen - Shannon Divergence at the optimal point. It is the generative model without the likelihood. However, there were some issues - GAN is very hard to train and it is very unstable. There were many proposed solutions to these above-mentioned problems.

One of the breakthrough was WGAN paper (https://arxiv.org/abs/1701.07875). Rather than finding the equilibrium between two neural networks, WGAN paper tries to minimize the 1-Wasserstein Distance(WD) between two networks. Intuitively, WD is the cost function of moving one distribution to the another. As the neural network is powerful function approximator, WGAN finds the optimal transport from the sample to the real distribution.

However, the functions we derived from the WGAN need to meet 1-Lipschitz condition. WGAN-GP came up with one solution to impose the gradient penalty(GP) as the gradient we obtained from the point between the real data and the samples deviates from 1. This approach works quite well.


#### Implementation

I've implemented WGAN-GP for MNIST data set using PyTorch 1.3.1. I assume that GPUs are available for this implementation and it supports multiple GPUs. You can test by changing the hyperparameters. Sample images are saved for every epoch, and model parameters and losses are recorded periodically.

