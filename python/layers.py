from operator import ge
import keras

def get_activation(name):
    if name == "leaky_relu": return "leaky"
    return "relu"

@keras.saving.register_keras_serializable()
class Dense(keras.layers.Dense):
  def export(self):
    act = get_activation(self.activation.__name__)
    return f""""\
[connected]
output={self.units}
activation={act}

"""


@keras.saving.register_keras_serializable()
class Conv2DSynabro(keras.layers.Conv2D):
  def export(self):
    act = get_activation(self.activation.__name__)
    return f""""\
[convolutional]
filters={self.filters}
size={self.kernel_size[0]}
stride={self.strides[0]}
pad={self.padding[0][1]}
activation={act}

"""

@keras.saving.register_keras_serializable()
class MaxPooling2D(keras.layers.MaxPooling2D):
  def export(self):
    
    return f"""\
[maxpool]
size={self.pool_size}
stride={self.strides[0]}

"""


@keras.saving.register_keras_serializable()
class Concatenate(keras.layers.Concatenate):
  def export(self):
    return f"""\
[route]
layers = -1, 8

"""


@keras.saving.register_keras_serializable()
class UpSampling2D(keras.layers.UpSampling2D):
  def export(self):
    return f"""\
[upsample]
stride={self.size[0]}

"""
