import keras
from converter import Converter

if __name__=="__main__":
  models = [
      ("yolov2_tiny", "./yolov2_tiny.keras"),
  ]
  for model_name, model_path in models:
    converter = Converter(model_name, model_path)
    converter.optimize()
    converter.convert()
    converter.export()
