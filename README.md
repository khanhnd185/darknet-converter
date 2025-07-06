# Darknet Converter

This repo is forked from [Darknet](https://github.com/pjreddie/darknet).
This project aims to convert pretrained neural network models from various frameworks (Keras, Tensorflow, ONNX) to Darknet.

## Run Darknet model

Run a object detection model

```
$ ./darknet detect \
  cfg/yolov3.cfg \
  weights/yolov3.weights \
  data/dog.jpg
```

Run a image classifier model

```
$ ./darknet classifier predict \
  cfg/imagenet1k.data \
  cfg/darknet19.cfg \
  weights/darknet19.weights \
  data/dog.jpg
```