**Object detection** is a computer technology more related to Computer Vision and Image Processing that deals with detecting instances of semantic objects of a certain class, such as humans, buildings, or cars, in digital images and videos [[1]]. Well-researched domains of object detection include face detection, car detection and pedestrian detection.

The models which we are using here are YOLO v3 and YOLO v4, which comprises the state-of-the-art object detection system for the real-time scenario and it is amazingly accurate and fast. YOLO stands for "You Only Look Once". Compared to other region proposal classification networks (fast R-CNN) which perform detection on various region proposals and thus end up performing prediction multiple times for various regions in a image, Yolo architecture is more like FCNN (fully convolutional neural network) and passes the image (n x n) once through the FCNN and output is (m x m) prediction. This the architecture is splitting the input image in m x m grid and for each grid generation 2 bounding boxes and class probabilities for those bounding boxes. We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.

###Environment:

In order to implement this project, We have to exploit **Google Colab’s** resources and the model got trained on Colab using **GPU**. 




## References
<a id="1">[1]</a>
Dasiopoulou, S., Mezaris, V., Kompatsiaris, I., Papastathis, V., & Strintzis, M. (2005). Knowledge-assisted semantic video object detection. Circuits and Systems for Video Technology, IEEE Transactions on, 15, 1210 - 1224.

<a id="2">[2]</a>

