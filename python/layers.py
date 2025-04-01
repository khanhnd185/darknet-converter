from operator import ge
import keras

def get_activation(name):
    if isinstance(name, keras.layers.LeakyReLU): return "leaky"
    return "relu"

@keras.saving.register_keras_serializable()
class Dense(keras.layers.Dense):
  def export_cfg(self):
    act = get_activation(self.activation)
    return f""""\
# {self.name}
[connected]
output={self.units}
activation={act}

"""
  def export_weights(self, f):
    if self.bias:
      f.write(self.weights[1].numpy().tobytes())
    f.write(self.weights[0].numpy().tobytes())


@keras.saving.register_keras_serializable()
class DepthwiseConv2D(keras.layers.DepthwiseConv2D):
  def export_cfg(self):
    act = get_activation(self.activation)
    return f""""\
# {self.name}
[convolutional]
filters={self.filters}
#groups=
size={self.kernel_size[0]}
stride={self.strides[0]}
pad={self.padding[0][1]}
activation={act}

"""
  def export_weights(self, f):
    if self.bias:
      f.write(self.weights[1].numpy().tobytes())
    f.write(self.weights[0].numpy().tobytes())


@keras.saving.register_keras_serializable()
class Conv2D(keras.layers.Conv2D):
  def export_cfg(self):
    act = get_activation(self.activation)
    return f""""\
# {self.name}
[convolutional]
filters={self.filters}
size={self.kernel_size[0]}
stride={self.strides[0]}
#pad=
activation={act}

"""
  def export_weights(self, f):
    if self.bias:
      f.write(self.weights[1].numpy().tobytes())
    f.write(self.weights[0].numpy().tobytes())

@keras.saving.register_keras_serializable()
class MaxPooling2D(keras.layers.MaxPooling2D):
  def export_cfg(self):
    
    return f"""\
# {self.name}
[maxpool]
size={self.pool_size}
stride={self.strides[0]}

"""
  def export_weights(self, f):
    return


@keras.saving.register_keras_serializable()
class Concatenate(keras.layers.Concatenate):
  def export_cfg(self):
    return f"""\
# {self.name}
[route]
layers = -1, 8

"""
  def export_weights(self, f):
    return


@keras.saving.register_keras_serializable()
class UpSampling2D(keras.layers.UpSampling2D):
  def export_cfg(self):
    return f"""\
# {self.name}
[upsample]
stride={self.size[0]}

"""
  def export_weights(self, f):
    return