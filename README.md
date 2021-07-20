**Object detection** is a computer technology more related to Computer Vision and Image Processing that deals with detecting instances of semantic objects of a certain class, such as humans, buildings, or cars, in digital images and videos [[1]]. Well-researched domains of object detection include face detection, car detection and pedestrian detection.

![alt text](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/Images/object_detection_on_roads.jpeg "Object Detection on Roads")
Object Detection On Roads [*ImageSource*](https://en.wikipedia.org/wiki/File:Computer_vision_sample_in_Sim%C3%B3n_Bolivar_Avenue,_Quito.jpg)

The models which we are using here are YOLO v3 and YOLO v4, which comprises the state-of-the-art object detection system for the real-time scenario and it is amazingly accurate and fast. YOLO stands for "You Only Look Once". Compared to other region proposal classification networks (fast R-CNN) which perform detection on various region proposals and thus end up performing prediction multiple times for various regions in a image, Yolo architecture is more like FCNN (fully convolutional neural network) and passes the image (n x n) once through the FCNN and output is (m x m) prediction. This the architecture is splitting the input image in m x m grid and for each grid generation 2 bounding boxes and class probabilities for those bounding boxes. We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.

### Environment:

In order to implement this project, We have to exploit **Google Colab’s** resources and the model got trained on Colab using **GPU**. 

![alt text](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/Images/GPU-version.png "GPU version")

The tabel below shows an overview of python lybraries we are using for these tasks.

| Distribution Name | Version   |       Description          |
| -------------     |:---------:| --------------------------:|
| Python            | 3.7       | >Python is an interpreted, object-oriented, high-level programming language used in many software development practices. |
| Pandas            | 1.1.2     | >Pandas is a software library, developed in python programming to handle operations on tabular data. It includes operations such as data manipulations, transformations and numeric analysis. |
| Numpy             | 1.19      | >Numpy is a python library that supports different operations to perform on multi dimensional arrays and matrices. |
| Opencv            | 4.5.2.54  | >OpenCV-Python is a library of Python bindings designed to solve computer vision problems. |
| Tensorflow        | 2.3.0     | >Tensorflow is an open source deep learning platform designed by google to support deep learning techniques like CNN, RNN, etc. |
| Keras             | 2.4       | >Keras is an open source library that acts as an API for developing deep learning applications and it runs on top of tensorflow. |
| Matplotlib        | 3.3.1     | >Matplotlib is a plotting library written in python framework. This library is helpful in creating static, animated and interactive visualizations in python. |
| Cuda              | 11.2      | >CUDA is a parallel computing platform and application programming interface model created by Nvidia. It allows software developers and software engineers to use a CUDA-enabled graphics processing unit for general purpose processing – an approach termed GPGPU. |

### Dataset:



## References
<a id="1">[1]</a>
Dasiopoulou, S., Mezaris, V., Kompatsiaris, I., Papastathis, V., & Strintzis, M. (2005). Knowledge-assisted semantic video object detection. Circuits and Systems for Video Technology, IEEE Transactions on, 15, 1210 - 1224.

<a id="2">[2]</a>

