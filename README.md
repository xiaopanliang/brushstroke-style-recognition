# brushstroke-style-recognition
The GitHub repository for the research project:

# Install
As admin or in venv:

```
pip install tensorflow
pip install scipy
pip install opencv-python # for cv2
```

## For DGX Docker image

Must run 
```
apt-get install -y libsm6 libxext6 libxrender-dev
```
in order for the import cv2 command to work (in addition to running the pip install above)

# Download
From
https://www.kaggle.com/teksab/imagenetvggverydeep19mat/downloads/imagenetvggverydeep19mat.zip/1
Download
imagenet-vgg-verydeep-19.mat
(Free account creation required)

***Obtain the dataset from someone...***

Run in this order:
* separation.py -- latest version uses style directory as input, but just one output directory. creaets sub-directory for you.
* make_dataset.py
* style_recognition.py

