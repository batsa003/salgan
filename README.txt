PyTorch Implementation of Generating Videos with Scene Dynamics.
=====================================

This repository contains a PyTorch implementation of [SalGAN: Visual Saliency Prediction with Generative Adversarial Networks](http://web.mit.edu/vondrick/tinyvideo/) by Carl Vondrick, Hamed Pirsiavash, Antonio Torralba, appeared at NIPS 2016. The model learns to generate tiny videos using adversarial networks.

I hope you find this implementation useful.

Results
-------------------


Example Generations
-------------------


Training
--------
The code requires a pytorch installation. 

Before you train the model, preprocess the dataset by running *preprocess.py* to resize the ground truth images and saliency maps to 256x192. 

To train the model, refer to main.py.

Data
----
We used the SALICON dataset for training. 

Reference:
---------
http://carlvondrick.com/tinyvideo/

