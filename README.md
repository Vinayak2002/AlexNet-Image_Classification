# Image Classification using AlexNet CNN

# Technologies Used:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) <br />
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)  <br />
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) <br />
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) <br />
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) <hr />

<h1><b>Problem Statement</b></h1>
<hr />
<i>PPG Dataset of 219 people is given, use any of the DNNs to classify the 2D images of the spectograms acquired from the given data.</i>
<hr />
<h3>Task:</h3>
<hr />
Convert the signal obtained from samples into 2-D spectrograms and classify the obtained images using pre-trained CNN models (like Alexnet, Resnet, Mobilenet etc.).
<hr />

### The CNN (Convolutional Neural Network) used in this project is called <a href="https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf">AlexNet</a>.

# Methodology

* Given PPG Signal -
<p align ="center" >
<img  width="700" src="https://github.com/Vinayak2002/AlexNet-Image_Classification/blob/main/images/PPG.png">
</p>

* PPG Signal Processing -
<p align ="center" >
<img  width="700" src="https://github.com/Vinayak2002/AlexNet-Image_Classification/blob/main/images/PPPG.png">
</p>

* Sectioning of PPG signal for data balancing -
<p align ="center" >
<img  width="700" src="https://github.com/Vinayak2002/AlexNet-Image_Classification/blob/main/images/SPPG.png">
</p>

* Spectogram of Sectioned Signal -
<p align ="center" >
<img  width="700" src="https://github.com/Vinayak2002/AlexNet-Image_Classification/blob/main/images/2image31.png">
</p>

* Normal Condition Spectogram -
<p align ="center" >
<img  width="700" src="https://github.com/Vinayak2002/AlexNet-Image_Classification/blob/main/images/0image10.png">
</p>

* Prehyptersion Condition Spectogram -
<p align ="center" >
<img  width="700" src="https://github.com/Vinayak2002/AlexNet-Image_Classification/blob/main/images/1image132.png">
</p>

* Stage-1 Hypertension Spectogram -
<p align ="center" >
<img  width="700" src="https://github.com/Vinayak2002/AlexNet-Image_Classification/blob/main/images/2image31.png">
</p>

* Stage-2 Hypertension Spectogram -
<p align ="center" >
<img  width="700" src="https://github.com/Vinayak2002/AlexNet-Image_Classification/blob/main/images/2image31.png">
</p>

<hr />
The spectogram dataset is then used for training and testing of the CNN model.
The model got overfitted since the data size was not adequate. The dataset was custom, and still being updated.
<hr />
* Results -
<p align ="center" >
<img  width="700" src="https://github.com/Vinayak2002/AlexNet-Image_Classification/blob/main/images/Result.png">
</p>
