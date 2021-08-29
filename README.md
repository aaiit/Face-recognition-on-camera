# Face-recognition-on-camera

1. Download the model weights  for vggFace from [google drive](https://drive.google.com/file/d/100fPG0cIa0GCKdwep8CDQcCv6I0lm1YU/view?usp=sharing)


2. Download the model wieght for face detection
```Bash
wget -N https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt


wget -N https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

```

3. Change config.json file

4. Run the code
```Python
py camera.py

```