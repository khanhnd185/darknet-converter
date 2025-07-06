import tensorflow as tf
import keras
import numpy as np
import PIL.Image, PIL.ImageOps
import os
import glob
import math
import abc
import tqdm
from labels import LABEL_COCO
from visualize import visualize_image

class ObjectDetectorBase:
  """Object detector base implementation."""

  def __init__(self,
    model_name,                         # model name as identifier
    model,                              # model or model file
    input_shape=None,                   # input shape
    dataset="coco",                     # dataset identifier
    keep_aspectratio=True,              # image: keep aspect ratio?
    preproc="raw",                      # image: preprocessor type
    score_thresh=0.30,                  # NMS: score threshold
    iou_thresh=0.60,                    # NMS: IOU(intersection-over-union) threshold
    max_ndetects=100,
    work_dir="./output"                 # working directory
  ):
    ##  Load a model if the model file is provided.
    if   (type(model)==str):
      extension = model.split(".")[-1]
      if   (extension in ["keras","h5","hdf5"]):
        model = keras.models.load_model(model)
      elif (extension=="tflite"):
        model = tf.lite.Interpreter(model)
      else: raise ValueError(f"Unknown model '{model}' provided")

    ##  Prepare for inferencing Keras and TFLite models.
    if   (isinstance(model, keras.Model)):
      self.model_type     = "keras"
      self.model          = model
      self.model_runner   = model.predict
      self.output_names   = model.output_names

      if (None in model.input_shape[-3:-1]):
        print(f"[I] model '{model_name}' has not enough information on input_shape")
        assert  input_shape!=None, "input_shape must provided"
        self.input_shape  = input_shape
      else:
        self.input_shape  = model.input_shape[-3:]
      print(f"[I] ObjectDetector({model_name}) created for Keras model.\n")
    elif (isinstance(model, tf.lite.Interpreter)):
      self.model_type     = "tflite"
      self.model          = model
      self.model_runner   = model.get_signature_runner()
      self.input_details  = self.model_runner.get_input_details()
      self.output_details = self.model_runner.get_output_details()
      self.input_name     = list(self.input_details.keys())[0]
      self.input_shape    = self.input_details[self.input_name]["shape"]
      self.output_names   = list(self.output_details.keys())
      print(f"[I] ObjectDetector({model_name}) created for TFLite model.\n")
    else: raise ValueError(f"Unknown model of '{type(model)}' provided")

    self.model_name       = model_name
    self.dataset          = dataset
    self.labels           = LABEL_COCO
    self.keep_aspectratio = keep_aspectratio
    self.preproc          = preproc
    self.score_thresh     = score_thresh
    self.iou_thresh       = iou_thresh
    self.max_ndetects     = max_ndetects
    self.work_dir         = work_dir
    os.makedirs(os.path.join(work_dir), exist_ok=True)


  def __repr__(self):
    return  f"""
  model_name      : {self.model_name}
  model_type      : {self.model_type}
  model           : {self.model}
  dataset         : {self.dataset}
  input_shape     : {str(tuple(self.input_shape)).replace(" ","")}
  keep_aspectratio: {self.keep_aspectratio}
  preproc         : {self.preproc}
  score_thresh    : {self.score_thresh}
  iou_thresh      : {self.iou_thresh}
  work_dir        : {self.work_dir}"""


  def __str__(self):
    return  self.__repr__()


  def preprocess(self,
    image_file                          # image file
  ):
    ##  Resize the image.
    _image  = PIL.Image.open(image_file).convert("RGB")
    XH,XW   = self.input_shape[-3:-1]   # input height, width

    if (self.keep_aspectratio):
      IW,IH = _image.size
      if (IH>IW): frame = [(IH-IW)//2, 0, IW, IH]
      else:       frame = [0, (IW-IH)//2, IW, IH]

      IS    = max(IH,IW)
      kS    = max(IS/XH,IS/XW)
      IH,IW = IS,IS
      kH,kW = kS,kS
      """
      image = PIL.ImageOps.pad(_image, (int(kS*XW),int(kS*XH)), color=(114,114,114), centering=(0.5,0.5))
      """
      image = PIL.ImageOps.pad(_image, (int(kS*XW),int(kS*XH)), color=(114,114,114), centering=(0,0))
      frame[0]  = 0
      frame[1]  = 0
    else:
      IW,IH = _image.size
      kH,kW = IH/XH,IW/XW
      frame = [0, 0, IW, IH]
      image = _image

    inputs  = np.asarray(image.resize((XW,XH)), dtype="float32")

    ##  Preprocess the input.
    if   (self.preproc=="stats" ):      # standardize statistics
      inputs  = ((inputs-[123.675,116.280,103.530])/[58.395,57.120,57.375]).astype(np.float32)
    elif (self.preproc=="minmax"):      # zero-center in (-1,+1] range
      inputs  = (inputs-127)/128
    elif (self.preproc=="raw"):         # raw RGB-8 in [0,+1) range
      inputs  =  inputs/256

    self._frame = frame                 # visible frame in padded image
    self._IH    = IH                    # warping image height
    self._IW    = IW                    # warping image width
    self._kH    = kH                    # warping image ratio in height
    self._kW    = kW                    # warping image ratio in width
    return  np.expand_dims(inputs,0), image


  @abc.abstractmethod
  def inference(self,
    image_file,                         # image file
    draw=True                           # draw overlaid image?
  ):  pass


  def inference_images(self,
    image_dir,                          # directory containing images
    nimages=None,                       # # of images to inference
    draw=True,                          # draw overlaid image?
    **kwargs
  ):
    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))
    if not(nimages):  nimages = len(image_files)

    print(f"[I] Inference {nimages} images in {image_dir}")
    for image_file in tqdm.tqdm(image_files[:nimages]):
      self.inference(image_file, draw=draw)
    print(f"Annotated images saved in {self.work_dir}")


  def postprocess_nms(self,
    scores,                             # scores  filtered (fast-NMS, score_threshold)
    classes,                            # classes filtered (fast-NMS, score_threshold)
    boxes                               # boxes in [x0,y0,x1,y1,w,h] cooridinates
  ):
    boxes_valid   = [True]*len(boxes)           # initialize all boxes valid
    boxes_area    = [_[4]*_[5] for _ in boxes]
    score_orders  = np.argsort(scores)[::-1]    # set the NMS order

    for i, id_i in enumerate(score_orders):
      if not(boxes_valid[id_i]):  continue

      for id_j in score_orders[i+1:]:
        if (not(boxes_valid[id_j]) or classes[id_i]!=classes[id_j]): continue

        x0  = max(boxes[id_i][0], boxes[id_j][0])
        y0  = max(boxes[id_i][1], boxes[id_j][1])
        x1  = min(boxes[id_i][2], boxes[id_j][2])
        y1  = min(boxes[id_i][3], boxes[id_j][3])
        if (x1<=x0 or y1<=y0):  continue

        area_ixj  = (x1-x0)*(y1-y0)
        area_iuj  = boxes_area[id_i]+boxes_area[id_j]-area_ixj
        if (area_ixj>=self.iou_thresh*area_iuj):  boxes_valid[id_j] = False

    ##  Detection outputs.
    detects     = []
    for id_i in score_orders:
      if (boxes_valid[id_i]): detects.append(id_i)
      if (len(detects)==self.max_ndetects): break

    return [
      np.array(scores )[detects],       # post-NMS scores
      np.array(classes)[detects],       # post-NMS classes
      np.array(boxes  )[detects]        # post-NMS boxes in [x0,y0,x1,y1,w,h] format
    ]


  def visualize(self,
    image,                              # PIL Image object
    scores,                             # list of scores
    classes,                            # list of classes
    boxes,                              # list of boxes
    box_format,                         # box coordinates format
    use_normalized_coordinates=True,    # coordinates normalized?
    max_boxes_to_draw=20,               # # of boxes to draw
    line_thickness=1,                   # line thickness
    label_fontsize=10                   # label font size
  ):
    return visualize_image(
      image, scores, classes, boxes, self.labels,
      box_format=box_format,
      use_normalized_coordinates=use_normalized_coordinates,
      max_boxes_to_draw=max_boxes_to_draw,
      line_thickness=line_thickness,
      label_fontsize=label_fontsize
    )

  def detect(self, image, draw=True):
    scores,classes,boxes = self.inference(image, draw=draw)
    print("  score   class            box coordinates ")
    print("  -----------------------------------------")
    for i, box in enumerate(boxes):
        print("  %.2f    %-12s    [%3d,%3d,%3d,%3d]" %(
        scores[i],
        self.labels[classes[i]]["name"],
        box[0], box[1], box[2], box[3]    # x0,y0,x1,y1
        ))

  @staticmethod
  def sigmoid(x): return 1/(1+np.exp(-x))

  @staticmethod
  def logit(x):   return 2*np.arctanh(2*x-1)


class ObjectDetectorYolo(ObjectDetectorBase):

  def __init__(self,
    model_name,                         # model name as identifier
    model,                              # model: Keras, TFLite
    input_shape=None,                   # input shape
    dataset="coco",                     # dataset identifier
    keep_aspectratio=True,              # image: keep aspect ratio?
    preproc="raw",                      # image: preprocessor type
    score_thresh=0.30,                  # NMS: score threshold
    iou_thresh=0.60,                    # NMS: IOU(intersection-over-union) threshold
    max_ndetects=100,
    work_dir="./scratch"                # working directory
  ):
    super().__init__(
      model_name, model,
      input_shape=input_shape,
      dataset=dataset,
      keep_aspectratio=keep_aspectratio,
      preproc=preproc,
      score_thresh=score_thresh,
      iou_thresh=iou_thresh,
      max_ndetects=max_ndetects,
      work_dir=work_dir
    )

    ##  Reorder TFLite outputs to match with the orignal.
    if (self.model_type=="tflite"):
      _details  = list(self.output_details.items())
      _details.sort(key=lambda _: ( -np.prod(_[1]["shape"][-3:-1]),-_[1]["shape"][-1]))
      self.output_names = [_[0] for _ in _details]

    ##  Extract the detection head information.
    if   (model_name in ["yolov2","yolov2_tiny"]):
      _headsplit  = (len(self.output_names)==2)
      _anchors    = 5
    elif (model_name in ["yolov3_tiny","yolov4_tiny"]):
      _headsplit  = (len(self.output_names)==4)
      _anchors    = 3
    elif (model_name=="yolov3"):
      _headsplit  = (len(self.output_names)==6)
      _anchors    = 3
    elif (model_name.startswith("yolov5") or model_name.startswith("yolov7")):
      _headsplit  = None
      _anchors    = 3
    elif (model_name.startswith("yolox")):
      _headsplit  = None
      _anchors    = 1
    else: raise NotImplementedError(f"Unknown model_name '{model_name}' provided")

    if   (self.model_type=="keras" ):
      _outputs        = self.model.outputs

      if (_headsplit):
            _nclasses5 = (_outputs[0].shape[-1]+_outputs[1].shape[-1])//_anchors
      else: _nclasses5 = (_outputs[0].shape[-1]                      )//_anchors
    elif (self.model_type=="tflite"):
      _output_details = self.model_runner.get_output_details()

      if (_headsplit):
            _nclasses5 = (_output_details[self.output_names[0]]["shape"][-1] +
                          _output_details[self.output_names[1]]["shape"][-1])//_anchors
      else: _nclasses5 =  _output_details[self.output_names[0]]["shape"][-1] //_anchors
    else: raise NotImplementedError(f"Unknown model_type '{self.model_type}' provided")

    self.nclasses   = _nclasses5-5      # number of classes
    self.headsplit  = _headsplit        # head split to (nclasses+3, 2)?


  def inference(self,
    image_file,                         # raw image file
    draw=True                           # draw?
  ):
    ##  Preprocess the input image.
    inputs,image  = self.preprocess(image_file)

    ##  Inference.
    if   (self.model_type=="keras"):
      outputs    = self.model_runner(inputs, verbose=0)
    elif (self.model_type=="tflite"):
      _outputs  = self.model_runner(**{self.input_name:inputs})
      outputs   = [_outputs[_].squeeze() for _ in self.output_names]
    else: raise ValueError(f"Unknown model_type '{self.model_type}' provided")

    ##  Postprocess the output tensor.
    if (self.headsplit):
      predicts  = [
        [np.reshape(_, (-1,self.nclasses+3)) for _ in outputs[ ::2]],
        [np.reshape(_, (-1,              2)) for _ in outputs[1::2]]
      ]
    else:
      predicts  = [np.reshape(_, (-1,self.nclasses+5)) for _ in outputs]

    if   (self.model_name in ["yolov2_tiny","yolov2"]):
      _postprocess = self.postprocess_yolov2
    elif (self.model_name in ["yolov3_tiny","yolov3","yolov4_tiny"]):
      _postprocess = self.postprocess_yolov3v4
    elif (self.model_name.startswith("yolov5") or self.model_name.startswith("yolov7")):
      _postprocess = self.postprocess_yolov5v7
    elif (self.model_name.startswith("yolox")):
      _postprocess = self.postprocess_yolox
    else: raise ValueError(f"Unknown model_name='{self.model_name}' provided")
    scores,classes,boxes  = _postprocess(predicts)

    ##  Report the output.
    if (draw):
      file_id     = image_file.split("/")[-1].split(".")[0]
      self.visualize(
        image, scores, classes, boxes, "x0y0x1y1wh",
        use_normalized_coordinates=False
      ).save(f"{self.work_dir}/{file_id}.png")
    return  scores,classes,boxes


  def postprocess_yolov2(self,
    predicts                            # predictions
  ):
    GH      = self.input_shape[-3]//32  # grid height
    GW      = self.input_shape[-2]//32  # grid width
    anchors = [
      [ 18.3274, 21.6763],
      [ 59.9827, 66.0010],
      [106.8298,175.1789],
      [252.2502,112.8890],
      [312.6566,293.3850]
    ]
    nboxes  = GH*GW*5                   # number of boxes total

    ##  Decode the coordinates for boxes exceeding the score threshold.
    scores,classes,boxes  = [],[],[]

    if (self.headsplit):                # header splitted to (nclasses+3, 2)
      for boxid in range(nboxes):
        P0      = predicts[0][boxid]    # encoded x,y and probabilities
        P1      = predicts[1][boxid]    # encoded w,h

        if (P0[2]>self.score_thresh):   # confidence P0[2]
          classid = np.argmax(P0[3:])   # find the top-1 class
          score   = P0[2]*P0[3+classid]
          a       =  boxid% 5           # anchor index
          xg      = (boxid//5)% GW      # grid reference xg
          yg      = (boxid//5)//GW      # grid reference yg
          xc      = ((  xg +P0[0])*32           )*self._kW -self._frame[0]
          yc      = ((  yg +P0[1])*32           )*self._kH -self._frame[1]
          w       = (np.exp(P1[0])*anchors[a][0])*self._kW
          h       = (np.exp(P1[1])*anchors[a][1])*self._kH
          boxes  .append([              # [x0,y0,x1,y1,w,h] format
            max(xc-w/2, 0),
            max(yc-h/2, 0),
            min(xc+w/2, self._frame[2]),
            min(yc+h/2, self._frame[3]),
                   w,
                   h
          ])
          classes.append(classid)
          scores .append(score  )
    else:
      for boxid in range(nboxes):
        P       = predicts[0][boxid]

        if (P[4]>__class__.logit(self.score_thresh)):   # confidence P[4]
          classid = np.argmax(P[5:])    # find the top-1 class
          score   = __class__.sigmoid(P[4])*__class__.sigmoid(P[5+classid])
          a       =  boxid% 5           # anchor index
          xg      = (boxid//5)% GW      # grid reference xg
          yg      = (boxid//5)//GW      # grid reference yg
          xc      = ((xg +__class__.sigmoid(P[0]))*32)*self._kW -self._frame[0]
          yc      = ((yg +__class__.sigmoid(P[1]))*32)*self._kH -self._frame[1]
          w       = (np.exp(P[2])*anchors[a][0]      )*self._kW
          h       = (np.exp(P[3])*anchors[a][1]      )*self._kH
          boxes  .append([              # [x0,y0,x1,y1,w,h] format
            max(xc-w/2, 0),
            max(yc-h/2, 0),
            min(xc+w/2, self._frame[2]),
            min(yc+h/2, self._frame[3]),
                   w,
                   h
          ])
          classes.append(classid)
          scores .append(score  )

    return  self.postprocess_nms(scores, classes, boxes)


  def postprocess_yolov3v4(self, predicts):
    if   (self.model_name=="yolov3"):
      YOLO_ANCHORS  = [
        [[ 10, 13], [ 16, 30], [ 33, 23]],
        [[ 30, 61], [ 62, 45], [ 59,119]],
        [[116, 90], [156,198], [373,326]]
      ]
    elif (self.model_name in ["yolov3_tiny","yolov4_tiny"]):
      if (self.dataset=="keti"):
        YOLO_ANCHORS  = [
          [[ 10, 13], [ 23, 27], [ 37, 58]],
          [[ 81, 82], [135,169], [344,319]]
        ]
      else:
        YOLO_ANCHORS  = [
          [[ 23, 27], [ 37, 58], [ 81, 82]],
          [[ 81, 82], [135,169], [344,319]]
        ]
    else: raise ValueError(f"unknown model_name='{self.model_name}' provided")

    scores,classes,boxes  = [],[],[]

    for p, anchors in enumerate(YOLO_ANCHORS):
      ##  Configure hyperparameters for each pyramid level.
      if (p==0):
        stride    = 8 if (len(YOLO_ANCHORS)==3) else 16
        GH        = self.input_shape[-3]//stride  # grid height
        GW        = self.input_shape[-2]//stride  # grid width
        nboxes    = GH*GW*3                   # number of boxes total
      else:
        stride  <<= 1
        GH      >>= 1
        GW      >>= 1
        nboxes  >>= 2

      ##  Decode the coordinates for boxes exceeding the score threshold.
      if (self.headsplit):              # header splitted to (nclasses+3, 2)
        for boxid in range(nboxes):
          P0      = predicts[0][p][boxid]   # encoded x,y and probabilities
          P1      = predicts[1][p][boxid]   # encoded w,h

          if (P0[2]>self.score_thresh):     # confidence P0[2]
            classid = np.argmax(P0[3:]) # find the top-1 class
            score   = P0[2]*P0[3+classid]
            a       =  boxid% 3         # anchor index
            xg      = (boxid//3)% GW    # grid reference xg
            yg      = (boxid//3)//GW    # grid reference yg
            xc      = ((  xg +P0[0])*stride       )*self._kW -self._frame[0]
            yc      = ((  yg +P0[1])*stride       )*self._kH -self._frame[1]
            w       = (np.exp(P1[0])*anchors[a][0])*self._kW
            h       = (np.exp(P1[1])*anchors[a][1])*self._kH
            boxes  .append([            # [x0,y0,x1,y1,w,h] format
              max(xc-w/2, 0),
              max(yc-h/2, 0),
              min(xc+w/2, self._frame[2]),
              min(yc+h/2, self._frame[3]),
                     w,
                     h
            ])
            classes.append(classid)
            scores .append(score  )
      else:
        for boxid in range(nboxes):
          P       = predicts[p][boxid]

          if (P[4]>__class__.logit(self.score_thresh)):   # confidence P[4]
            classid = np.argmax(P[5:])  # find the top-1 class
            score   = __class__.sigmoid(P[4])*__class__.sigmoid(P[5+classid])
            a       =  boxid% 3         # anchor index
            xg      = (boxid//3)% GW    # grid reference xg
            yg      = (boxid//3)//GW    # grid reference yg
            xc      = ((xg +__class__.sigmoid(P[0]))*stride)*self._kW -self._frame[0]
            yc      = ((yg +__class__.sigmoid(P[1]))*stride)*self._kH -self._frame[1]
            w       = (np.exp(P[2])*anchors[a][0]          )*self._kW
            h       = (np.exp(P[3])*anchors[a][1]          )*self._kH
            boxes  .append([            # [x0,y0,x1,y1,w,h] format
              max(xc-w/2, 0),
              max(yc-h/2, 0),
              min(xc+w/2, self._frame[2]),
              min(yc+h/2, self._frame[3]),
                     w,
                     h
            ])
            classes.append(classid)
            scores .append(score  )

    return  self.postprocess_nms(scores, classes, boxes)


  def postprocess_yolov5v7(self, predicts):
    if   (self.model_name.startswith("yolov5") or self.model_name=="yolov7_tiny"):
      YOLO_ANCHORS  = [
        [[ 10, 13], [ 16, 30], [ 33, 23]],
        [[ 30, 61], [ 62, 45], [ 59,119]],
        [[116, 90], [156,198], [373,326]]
      ]
    elif (self.model_name in ["yolov7","yolov7_x"]):
      YOLO_ANCHORS  = [
        [[ 12, 16], [ 19, 36], [ 40, 28]],
        [[ 36, 75], [ 76, 55], [ 72,146]],
        [[142,110], [192,243], [459,401]]
      ]
    else: raise ValueError(f"unknown model_name='{self.model_name}' provided")

    scores,classes,boxes  = [],[],[]

    for p, anchors in enumerate(YOLO_ANCHORS):
      ##  Configure hyperparameters for each pyramid level.
      if (p==0):
        stride    = 8
        GH        = self.input_shape[-3]//stride  # grid height
        GW        = self.input_shape[-2]//stride  # grid width
        nboxes    = GH*GW*3                       # number of boxes total
      else:
        stride  <<= 1
        GH      >>= 1
        GW      >>= 1
        nboxes  >>= 2

      ##  Decode the coordinates for boxes exceeding the score threshold.
      for boxid in range(nboxes):
        P         = predicts[p][boxid]

        if (P[4]>self.score_thresh):    # confidence P[4]
          classid = np.argmax(P[5:])    # find the top-1 class
          score   = P[4]*P[5+classid]
          a       =  boxid% 3           # anchor index
          xg      = (boxid//3)% GW      # grid reference xg
          yg      = (boxid//3)//GW      # grid reference yg
          xc      = (xg+P[0]*2-0.5)*stride       *self._kW -self._frame[0]
          yc      = (yg+P[1]*2-0.5)*stride       *self._kH -self._frame[1]
          w       = (   P[2]*2)**2 *anchors[a][0]*self._kW
          h       = (   P[3]*2)**2 *anchors[a][1]*self._kH
          boxes.append([                # [x0,y0,x1,y1,w,h] format
            max(xc-w/2, 0),
            max(yc-h/2, 0),
            min(xc+w/2, self._frame[2]),
            min(yc+h/2, self._frame[3]),
                   w,
                   h
          ])
          classes.append(classid)
          scores .append(score  )

    return  self.postprocess_nms(scores, classes, boxes)


  def postprocess_yolox(self, predicts):
    if (len(predicts)==1):              # concatenated output
      predicts  = [predicts[0][:2704], predicts[0][2704:3380], predicts[0][3380:] ]

    scores,classes,boxes  = [],[],[]

    for p in range(len(predicts)):
      ##  Configure hyperparameters for each pyramid level.
      if (p==0):
        stride    = 8
        GH        = self.input_shape[-3]//stride  # grid height
        GW        = self.input_shape[-2]//stride  # grid width
        nboxes    = GH*GW                         # number of boxes total
      else:
        stride  <<= 1
        GH      >>= 1
        GW      >>= 1
        nboxes  >>= 2

      ##  Decode the coordinates for boxes exceeding the score threshold.
      for boxid in range(nboxes):
        P         = predicts[p][boxid]

        if (P[4]>self.score_thresh):    # confidence P[4]
          classid = np.argmax(P[5:])    # pick the top-1 class
          score   = P[4]*P[5+classid]
          xg      = boxid% GW           # grid reference xg
          yg      = boxid//GW           # grid reference yg
          xc      = (xg   +P[0])*stride*self._kW -self._frame[0]
          yc      = (yg   +P[1])*stride*self._kH -self._frame[1]
          w       = np.exp(P[2])*stride*self._kW
          h       = np.exp(P[3])*stride*self._kH
          boxes   .append([              # [x0,y0,x1,y1,w,h] format
            max(xc-w/2, 0),
            max(yc-h/2, 0),
            min(xc+w/2, self._frame[2]),
            min(yc+h/2, self._frame[3]),
                   w,
                   h
          ])
          classes .append(classid)
          scores  .append(score  )
    return  self.postprocess_nms(scores, classes, boxes)
