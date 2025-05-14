# Depixelation Tool

## Description
A machine learning tool designed to recognize text from heavily pixelated images. This tool generates datasets of pixelated images, trains a neural network model to recognize the strings, and provides benchmarking capabilities to evaluate the model's performance.

The model is built using Pytorch, leveraging convolutional neural networks (CNNs) and long short-term memory networks (LSTM) with connectionist temporal classification (CTC) loss for sequence prediction. Inputs are converted to monochrome.
