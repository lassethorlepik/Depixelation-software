# Depixelation Tool

![example](sources/example.png)

## Description
A machine learning tool designed to recognize text from heavily pixelated images. This tool generates datasets of pixelated images, trains a neural network model to recognize the strings, and provides benchmarking capabilities to evaluate the model's performance.

The model is built using Pytorch, leveraging convolutional neural networks (CNNs) and long short-term memory networks (LSTM) with connectionist temporal classification (CTC) loss for sequence prediction. Inputs are converted to monochrome.

## Usage
Launch setup.py to install requirements in venv.
Start launcher.py after setup is done.

### Dataset Generation
Edit `synthesizer/data_synth.py` to specify the configuration.
Generate a dataset of pixelated images by running `launch.py` and choosing the "Synthesize images" option.

### Training a model
Edit `config.json` to specify the training configuration.
Run `launch.py` and choose the "Train model" option.

### Licensing

This project is licensed under the [MIT license](LICENSE).
