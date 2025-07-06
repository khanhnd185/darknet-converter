import keras
from converter import (
    Keras2Darknet
  , Darknet2Keras
)
from detector import ObjectDetectorYolo

if __name__=="__main__":
  test_image = "../data/dog.jpg"
  output_dir = "./output"

  for model_name, model_path in [
    [
      "yolov2_tiny" , [
          "../weights/yolov2_tiny.cfg"
        , "../weights/yolov2_tiny.weights"
      ]
    ]
    , [
      "yolov3_tiny" , [
          "../cfg/yolov3-tiny.cfg"
        , "../weights/yolov3-tiny.weights"
      ]
    ]
  ]:
    d2k = Darknet2Keras(
        model_name
      , model_path
      , [416, 416, 3]
      , output_dir=output_dir
    )
    keras_model = d2k.convert()
    keras_model.save(f"{output_dir}/{model_name}.keras")

    detector = ObjectDetectorYolo(
        model_name
      , keras_model
      , score_thresh=0.3
      , iou_thresh=0.65
      , work_dir=output_dir
    )
    detector.detect(test_image)

    k2d = Keras2Darknet(model_name, f"{output_dir}/{model_name}.keras")
    k2d.optimize()
    k2d.export()
    k2d.convert()
