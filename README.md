# SEP-Emotion-Detection-CNN
This project implements a CNN from scratch for emotion classification on the FERPlus dataset.

This project requires python 3.11. 

We suggest creating a python virtual environment.

Environmental setup:

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

----------------------------------------------------------------------

If you have a nvidia gpu, gpu acceleration is available for CUDA 12.1.

If not each script will automatically fall back to cpu usage.

----------------------------------------------------------------------

Training and evaluation was solely done on FERPlus

Dataset used for training: FERPlus -> https://www.kaggle.com/datasets/arnabkumarroy02/ferplus

The downloaded images should be put into the same folder that the scripts are in.

Preprocessing of images is done in each script.


-----------------------------------------------------------------------

ImageExampleScript.py -> Generates an image with Grad-CAM visualization.

Training.py -> Training script for the CNN

cnn.py -> CNN architecture

demo.py -> Webcam demo for live classification

evaluation.py -> Evaluation script (to test on other datasets, change root inside)

output.xlsx -> FERPlus test set results using the best weights

videodemo.py -> Classify emotions in a video

weights.pth -> Best trained model weights

