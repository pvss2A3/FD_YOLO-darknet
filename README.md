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
| Python            | 3.7       | Python is an interpreted, object-oriented, high-level programming language used in many software development practices. |
| Pandas            | 1.1.2     | Pandas is a software library, developed in python programming to handle operations on tabular data. It includes operations such as data manipulations, transformations and numeric analysis. |
| Numpy             | 1.19      | Numpy is a python library that supports different operations to perform on multi dimensional arrays and matrices. |
| Opencv            | 4.5.2.54  | OpenCV-Python is a library of Python bindings designed to solve computer vision problems. |
| Tensorflow        | 2.3.0     | Tensorflow is an open source deep learning platform designed by google to support deep learning techniques like CNN, RNN, etc. |
| Keras             | 2.4       | Keras is an open source library that acts as an API for developing deep learning applications and it runs on top of tensorflow. |
| Matplotlib        | 3.3.1     | Matplotlib is a plotting library written in python framework. This library is helpful in creating static, animated and interactive visualizations in python. |
| Cuda              | 11.2      | CUDA is a parallel computing platform and application programming interface model created by Nvidia. It allows software developers and software engineers to use a CUDA-enabled graphics processing unit for general purpose processing – an approach termed GPGPU. |

### Dataset:

At present, the published mask data sets are few, and there are problems such as poor content, poor quality and single background which cannot be directly applied to the face mask detection task in a complex environment. Under such context, this paper adopts the method of using my own photos and screening from the published [Face Mask Detection dataset from Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection?select=annotations) and we have downloaded it directly to our Google Drive. The dataset consists of two folders:

* **images**, which comprises 853 *'.png'* files
* **annotations**, which comprises 853 corresponding *'.xml'* annotations.

In the whole data set, there are 853 images of which 767 images are selected as training set and remaining 86 images are selected as testing/validation set. The images we have selected for training/testing sets need to be manually split into another two folders, one for training image data and the other for testing image data. In YOLO, the labelling format for any image data should be in *'.txt'* format. So for this, we need to convert our *'.xml'* files from *'annotations'* folder into *'.txt'* format.

To create a .txt file we need 5 things from each *.xml* file. For each `<object> ... <\object>`in an *.xml* file fetch the **class** (namely the `<name> ... <\name>` field), and the coordinates of the **bounding box** (namely the 4 attributes in `<bndbox> ... <\bndbox>`). The desirable format should look like as follows: 

`<class_name><x_centre><y_centre><width><height>`

To achieve above requirement, we need to use [yolo_xml_to_txt.py](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/yolo_xml_to_txt.py) script, which fetches the aforementioned 5 attributes for each object in each *.xml* file and creates the corresponding *.txt* files. For example, for an [image](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/Images/maksssksksss35.png) the associated .txt file is [sample txt](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/Images/maksssksksss35.txt). And this is the exact conversion of the an [.xml](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/Images/maksssksksss35.xml) file into a *.txt* file. 

After converting all the *.xml* files from annotaions folder to *.txt*, we have to copy the related *.txt* annotation file in the same folder where our image data is available. So, the training image data folder should contain 767 images and corresponding 767 *.txt* annotation files and the testing image data folder should contain rest 86 images with their corresponding 86 *.txt* annotation files. To check you have how many files in those folders you can use the following code:

```python
%cd (...your training set folder...)
!ls -F | grep .png | wc -l
!ls -F | grep .txt | wc -l
%cd ..

%cd (...your test folder...)
!ls -F | grep .png | wc -l
!ls -F | grep .txt | wc -l
%cd ..
```
You can also check the conversion was correct or not with [bb script](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/show_bb.py). The script takes an image and its corresponding .txt annotation from a given folder and displays the image with the ground truth bounding boxes.

### Training the dataset:

The next step is to train on our dataset. For this we need to clone the <ins> darknet repo </ins> [[2]] by running:
  
  `!git clone https://github.com/AlexeyAB/darknet`

and after that, we need to download the weights of the pre-trained model in order to apply transfer learning and not train the model from scratch. Here we are using `!wget https://pjreddie.com/media/files/darknet53.conv.74` for YOLOv3 (you can also use `!wget https://pjreddie.com/media/files/yolov3.weights` but *darknet53.conv.74* weights got more efficient results compared to *yolov3.weights*) and `!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137` for YOLOv4. 

**darknet53.conv.74** is the backbone of the YOLOv3 network which is originally trained for classification on the ImageNet dataset and plays the role of the extractor. So, 107 layers are loaded in our YOLOv3 model case and 162 layers in our YOLOv4 case.

Before training our model, we need to create some files such as *obj.names, obj.data, obj.cfg, train.txt* and *test.txt*. Below is some description about what should these files should contain. 

1. **obj.names**: 
  > create a file obj.names which contains the classes of the problem. In our case, the original Kaggle dataset has 3 categories: *with_mask, without_mask, and mask_weared_incorrect*. To simplify a little bit the task, we considered the two latter categories into one. Thus, for our task, we have two categories: ***mask*** and ***no_mask*** based on whether someone wears his/her mask appropriately. For both the models, we have used same [face_mask.names](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/YOLOv3/face_mask.names) file.
2. **obj.data**: 
  > create a obj.data file that includes relevant information (where classes = number of objects) to our problem and it is going to be used from the program. For YOLOv3, we have created [face_mask.data](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/YOLOv3/face_mask.data) and for YOLOv4, it will be [yolov4_face_mask.data](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/YOLOv4/yolov4_face_mask.data). Make sure you also have *backup* folder in the dataset folder, because in this *backup* folder the weights are going to be saved after every 1000 iterations. These will actually be your checkpoints in case of an unexpected interruption, from where you can continue the training process.
3. **obj.cfg**:
 > Create file obj.cfg with the same content as in yolo_custom_version.cfg files and:
    1. change line batch to `batch=64`
    2. change line subdivisions to `subdivisions=16`
    3. change line max_batches to (classes x 2000, but not less than number of training images and not less than 4000), f.e. max_batches=6000 if you train for 3 classes. In our case, number of classes are 2, so we will get max_batches as 2x2000=4000 but we are using `max_batches=7000` for YOLOv3 and `max_batch=4000` for YOLOv4 
    4. change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400. In our case `steps=5600,6300` for YOLOv3 and `steps=3200,3600` for YOLOv4
    5. set network size `width=416 height=416` or any value multiple of 32.
    6. change line classes=80 to your number of objects (in our case `classes = 2`) in each of 3 \[yolo\]-layers
    7. change \[filters=255\] to filters=(classes + 5)x3 (in our case `filters = 21`) in the 3 \[convolutional\] before each \[yolo] layer, keep in mind that it only has to be the last \[convolutional] before each of the \[yolo] layers.
 > The obj.cfg files for YOLOv3 and YOLOv4 are [face_mask.cfg](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/YOLOv3/face_mask.cfg) and [yolov4_face_mask.cfg](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/YOLOv4/yolov4_face_mask.cfg) respectively.
4. **train.txt & text.txt**:
 > These two files have been included in the obj.data file (in our case face_mask.data & yolov4_face_mask.data) and indicate the absolute path for each image to the model. Those files will look like [train.txt](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/resources/train.txt) & [text.txt](https://github.com/pvss2A3/FD_YOLO-darknet/blob/main/resources/test.txt)
  
After creating all the files, change the related permissions of the darknet folder by running `!chmod +x ./darknet` command and then copy *obj.data*, *obj.names*, *train.txt* & *test.txt* files in *darknet/data* folder and *obj.cfg* file in *darknet/cfg* folder. And finally, begin the training by 

   `!./darknet detector train data/face_mask.data cfg/face_mask.cfg darknet53.conv.74.1 -dont_show -i 0 -map -points 0` for YOLOv3 and
   `!./darknet detector train data/yolov4_face_mask.data cfg/yolov4_face_mask.cfg yolov4.conv.137 -dont_show -i 0 -map -points 0` for YOLOv4
The flag `-map` will inform us about the progress of the training by printing out important metrics such as average Loss, Precision, Recall, AveragePrecision (AP), meanAveragePrecsion (mAP), etc. The training process might take many hours depending on various parameters and your hardware setup performance, and it is normal. For this project, in order to train my models up to this point, I needed about 8.05 hours for YOLOv3 and 6.73 hours for YOLOv4. Suppose in case of any reason, model stops training after multiple of 1000 iterations, then you can continue the training process from that iterations by using partially trained model. For example, we stopped our YOLOv4 model training after 1000 iterations then we can resume this training process by running `!./darknet detector train data/yolov4_face_mask.data cfg/yolov4_face_mask.cfg /content/gdrive/MyDrive/FD_dataset/backup1/yolov4_face_mask_1000.weights -dont_show -i 0 -map -points 0` command. Make sure your weights are saved in the backup folder. If the model stops training before 1000 iterations, then we need to start training the model from the begining because the weights are saved only after every 1000 iterations.

After training the model is done then you can check for models mAP@0.5 by `!./darknet detector test data/obj.data cfg/obj.cfg backup/obj_best.weights` code. Our YOLOv3 and YOLOv4 models have acheived mAP@0.5 as 85.97% and 88.03% respectively.




## References
<a id="1">[1]</a>
Dasiopoulou, S., Mezaris, V., Kompatsiaris, I., Papastathis, V., & Strintzis, M. (2005). Knowledge-assisted semantic video object detection. Circuits and Systems for Video Technology, IEEE Transactions on, 15, 1210 - 1224.

<a id="2">[2]</a>
https://github.com/AlexeyAB/darknet

<a id="3">[3]</a>

