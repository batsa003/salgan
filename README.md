PyTorch Implementation of SalGAN: Visual Saliency Prediction with Generative Adversarial Networks
=====================================

This repository contains a PyTorch implementation of [SalGAN: Visual Saliency Prediction with Generative Adversarial Networks](https://github.com/imatge-upc/saliency-salgan-2017) by Junting Pan et al,. The model learns to predict a saliency map given an input image.

I hope you find this implementation useful.

Results
-------------------
TODO: 


Example Generations
-------------------
TODO: Predict.ipynb


Training
--------
The code requires a pytorch installation. 

Before you train the model, preprocess the dataset by running *preprocess.py* to resize the ground truth images and saliency maps to 256x192. 

To train the model, refer to main.py.

Pretrained Model
--------
TODO: 

Data
----
We used the SALICON dataset for training. 

Reference:
---------
https://imatge-upc.github.io/saliency-salgan-2017/

