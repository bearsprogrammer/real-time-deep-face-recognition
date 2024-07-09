# real-time-deep-face-recogniton

Real-time face recognition program using Google's facenet.
* [youtube video](https://www.youtube.com/watch?v=T6czH6DLhC4)
## Inspiration
* [OpenFace](https://github.com/cmusatyalab/openface)
* I refer to the facenet repository of [davidsandberg](https://github.com/davidsandberg/facenet).
* also, [shanren7](https://github.com/shanren7/real_time_face_recognition) repository was a great help in implementing.
## Dependencies
* Tensorflow 1.2.1 - gpu
* Python 3.5
* Same as [requirement.txt](https://github.com/davidsandberg/facenet/blob/master/requirements.txt) in [davidsandberg](https://github.com/davidsandberg/facenet) repository.
## Pre-trained models
* Inception_ResNet_v1 CASIA-WebFace-> [20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE/edit)
## Face alignment using MTCNN
* You need [det1.npy, det2.npy, and det3.npy](https://github.com/davidsandberg/facenet/tree/master/src/align) in the [davidsandberg](https://github.com/davidsandberg/facenet) repository.
## How to use
* First, we need align face data. So, if you run 'Make_aligndata.py' first, the face data that is aligned in the 'output_dir' folder will be saved.
* Second, we need to create our own classifier with the face data we created. <br/>(In the case of me, I had a high recognition rate when I made 30 pictures for each person.)
</br>Your own classifier is a ~.pkl file that loads the previously mentioned pre-trained model ('[20170511-185253.pb](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE/edit)') and embeds the face for each person.<br/>All of these can be obtained by running 'Make_classifier.py'.<br/>
* Finally, we load our own 'my_classifier.pkl' obtained above and then open the sensor and start recognition.
</br> (Note that, look carefully at the paths of files and folders in all .py)
### Docker
* First, clone this repo and change current directory to its directory on your device. Next, put the images of each person in separated directories with the names that you want to be shown for each person inside the '*data*' directory; for example, Daehyun's photos are in: '*./data/Daehyun*' and Byeonggil's are in: '*./data/Byeonggil*'. Then, you can run it in two ways: <br/>1. Build the Docker image by running: <br/>`docker build -t "your_image_name":"your_image_tag" .` <br/>After the build is done, run: <br/>`docker run --device /dev/video0:/dev/video0 "your_image_name":"your_image_tag` <br/>2. Run: <br/>`docker compose up`. <br/> (Note that Windows doesn’t give camera access to Docker containers. So, you can just run this container on Linux.)
## Result
<img src="https://github.com/bearsprogrammer/real-time-deep-face-recogniton/blob/master/realtime_demo_pic.jpg" width="60%">