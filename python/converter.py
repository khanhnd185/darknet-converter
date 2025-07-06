import os
import io
import keras
import tempfile
import collections
import configparser
import numpy as np
import layers as darknet_layer
import tensorflow as tf

from keras import Model
from keras.layers import *
from keras.src.layers.core.tf_op_layer import TFOpLambda

def build_model(
    inputs
  , outputs
  , name=None
):
  model       = keras.Model(inputs, outputs, name=name)
  model_file  = tempfile.mkstemp(suffix=".keras")[1]
  model.save(model_file)              # save-and-load the model for cleanup
  model       = keras.models.load_model(model_file)
  os.system(f"rm -rf {model_file}")
  return  model


class Keras2Darknet:
  def __init__(
        self
      , model_name
      , model_path
    ):

    self.convertion = {
      keras.layers.Conv2D          : darknet_layer.Conv2D,
      keras.layers.Dense           : darknet_layer.Dense,
      keras.layers.DepthwiseConv2D : darknet_layer.DepthwiseConv2D,
      keras.layers.Concatenate     : darknet_layer.Concatenate,
      keras.layers.MaxPool2D       : darknet_layer.MaxPooling2D,
    }
    self.model              = keras.models.load_model(model_path)
    self.model_name         = model_name
    self.darknet_model      = None
    self.infered_model      = None
    print(f"[I] Load model '{model_path}'.")

  def export(self):
    if not self.infered_model: return

    print(f"[I] Convert model '{self.model.name}'.")
    model   = self.infered_model
    SIGNALS = {}
    OUTPUTS = [_.node.outbound_layer for _ in model.outputs]

    for layer_id, L0 in enumerate(model.layers):
      if (len(L0.inbound_nodes)>1): raise NotImplementedError
      LN1 = L0.inbound_nodes [0].inbound_layers

      if   (hasattr(L0, "removed")):
        SIGNALS[L0] = SIGNALS[LN1]
      elif (isinstance(L0, InputLayer)):
        INPUTS      = Input(shape=model.input_shape[1:], dtype=L0.dtype, name="images")
        SIGNALS[L0] = INPUTS
      elif (isinstance(L0, tuple(self.convertion.keys()))):
        SIGNALS[L0] = self.convertion[type(L0)](**L0.get_config())(SIGNALS[LN1])
      else: raise ValueError(f"Unknown layer {L0.name} of type {type(L0)}")

    self.darknet_model = build_model(INPUTS, [SIGNALS[_] for _ in OUTPUTS], name=model.name)

  def convert(self):
    if not self.darknet_model: return

    print(f"[I] Export darknet cfg '{self.model_name}'.")
    os.makedirs(f"./{self.model_name}", exist_ok=True)
    content = f"# Model {self.model_name}\n\n"
    for layer in self.darknet_model.layers:
      if (isinstance(layer, InputLayer)): continue
      content += layer.export_cfg()
    open(f"./{self.model_name}/{self.model_name}.cfg","w").write(content)

    print(f"[I] Export darknet weights '{self.model_name}'.")
    major, minor, revision = 0, 2, 0
    f = open(f"./{self.model_name}/{self.model_name}.weights","wb")
    f.write(bytes([major, minor, revision]))
    #f.write(np.ndarray((1), dtype="int64", buffer=bytes([0])).tobytes())
    for layer in self.darknet_model.layers:
      if (isinstance(layer, InputLayer)): continue
      layer.export_weights(f)
    f.close()

  def optimize(
      self
    , fuse_preproc=False
    , preproc_fn=None
  ):
    model = self.model

    print(f"[I] Create the inference model '{model.name}'.")
    SIGNALS = {}
    OUTPUTS = [_.node.outbound_layer for _ in model.outputs]

    for layer_id, L0 in enumerate(model.layers):
      if (len(L0.inbound_nodes)>1): raise NotImplementedError
      LN1 = L0.inbound_nodes [0].inbound_layers
      L1  = L0.outbound_nodes[0].outbound_layer if (len(L0.outbound_nodes)>0) else None

      if   (hasattr(L0, "removed")):
        if (type(LN1)==list):
          if (isinstance(LN1[1], TFOpLambda) and (LN1[1].function.__name__ == 'multiply')):
            SIGNALS[L0] = SIGNALS[LN1[1]]
          else:
            SIGNALS[L0] = SIGNALS[LN1[0]]
        else:
          SIGNALS[L0] = SIGNALS[LN1]

      elif (isinstance(L0, InputLayer)):  # [InputLayer]
        INPUTS      = Input(shape=model.input_shape[1:], dtype=L0.dtype, name="images")
        SIGNALS[L0] = INPUTS
      elif (isinstance(L0, (Conv2D, DepthwiseConv2D))): # [Conv2D], [DepthwiseConv2D]
        use_bias    = L0.use_bias
        activation  = L0.activation

        if (isinstance(L1, BatchNormalization)):
          kernel      = L0.weights[0].numpy()
          bias        = L0.weights[1].numpy() if (use_bias)  else 0
          gamma       = L1.gamma.numpy()      if (L1.scale)  else 1
          beta        = L1.beta.numpy()       if (L1.center) else 0
          mean        = L1.moving_mean.numpy()
          std         = np.sqrt(L1.moving_variance+L1.epsilon)
          L1.removed  = True
          use_bias    = True

          if (isinstance(L0, Conv2D)):  k =                gamma/std*           kernel
          else:                         k = np.expand_dims(gamma/std*np.squeeze(kernel),-1)
          b  = beta + (gamma/std)*(bias-mean)
          LA = L1.outbound_nodes[0].outbound_layer if (len(L1.outbound_nodes)>0) else None

        else:                                     # Conv2D
          k  = L0.weights[0].numpy()
          b  = L0.weights[1].numpy() if (use_bias) else 0
          LA = L1

        if (fuse_preproc):
          f               = np.array([[0,0,0],[1,1,1]], dtype="float32")
          preproc_layers  = []

          while (True):
            if   (isinstance(LN1, InputLayer)): break
            elif (isinstance(LN1, (Rescaling, Normalization))):  preproc_layers.insert(0,LN1)
            LN1 = LN1.inbound_nodes[0].inbound_layers

          if   (len(preproc_layers)>0):
            for layer in preproc_layers:  f = layer(f)
            f   = np.squeeze(f)
          elif (preproc_fn!=None):
            f   = preproc_fn(f)
          else: raise ValueError(f"Invalid preproc_fn={preproc_fn} provided")

          b_shiftscale  = -f[0]
          k_scale       = (f[1]-f[0])*256
          k             = np.moveaxis(k, -1,  0)  # HWCN-to-NWHC
          b            -= np.sum(b_shiftscale*np.sum(k,axis=(1,2)), 1)
          k            *= k_scale
          k             = np.moveaxis(k,  0, -1)  # NHWC-to-HWCN
          use_bias      = True
          fuse_preproc  = False

        if   (L0.activation.__name__=="linear"):
          if (len(L0.outbound_nodes)==1 and isinstance(LA, (Activation, ReLU, LeakyReLU))):
            activation  = LA.activation if (isinstance(LA, Activation)) else LA
            LA.removed  = True

        kwargs  = {
          "kernel_size" : L0.kernel_size,
          "strides"     : L0.strides,
          "padding"     : L0.padding,
          "use_bias"    : use_bias,
          "activation"  : activation,
          "weights"     : [k,b] if (use_bias) else [k],
          "name"        : L0.name
        }
        if (isinstance(L0, Conv2D)):  _layer = Conv2D(L0.filters, **kwargs)
        else:                         _layer = DepthwiseConv2D(   **kwargs)

        SIGNALS[L0] = _layer(SIGNALS[LN1])
      elif (isinstance(L0, Dense)):
        if ((L0 in OUTPUTS) and (L0.activation.__name__=="softmax")):
          L0.activation = keras.activations.linear

        SIGNALS[L0] = L0(SIGNALS[LN1])
      elif (isinstance(L0, Activation)):
        if ((L0 in OUTPUTS) and (L0.activation.__name__=="softmax")):
          SIGNALS[L0] = SIGNALS[LN1]
        else:
          SIGNALS[L0] = L0(SIGNALS[LN1])
      elif (isinstance(L0, (Rescaling, Normalization))):
        if (fuse_preproc):  SIGNALS[L0] = SIGNALS[LN1]
        else:               SIGNALS[L0] = L0(SIGNALS[LN1])

      elif (isinstance(L0, Dropout)):   # [layers to remove]
        SIGNALS[L0] = SIGNALS[LN1]
      elif (isinstance(L0, TFOpLambda)):
        if ((L0.function.__name__ == '_add_dispatch') and
            (isinstance(L1, ReLU)) and
            (len(L1.outbound_nodes)>0) and
            (isinstance(L1.outbound_nodes[0].outbound_layer, TFOpLambda)) and
            (L1.outbound_nodes[0].outbound_layer.function.__name__ == 'multiply')):
          L2  = L1.outbound_nodes[0].outbound_layer
          L3  = L2.outbound_nodes[0].outbound_layer if (len(L2.outbound_nodes)>0) else None
          if ((isinstance(L3, Multiply)) and
              ((L3.inbound_nodes[0].inbound_layers[0] == LN1) or
              (L3.inbound_nodes[0].inbound_layers[1] == LN1))):
            SIGNALS[L0] = Activation('hard_swish')(SIGNALS[LN1])
            print(f"[{layer_id}] {L0.function.__name__ }: hard_swish")
            L1.removed  = True
            L2.removed  = True
            L3.removed  = True
          else:
            SIGNALS[L0] = Activation('hard_sigmoid')(SIGNALS[LN1])
            print(f"[{layer_id}] {L0.function.__name__ }: hardsigmoid")
            L1.removed  = True
            L2.removed  = True
        else:
          SIGNALS[L0] = L0([SIGNALS[_] for _ in LN1] if (type(LN1)==list) else SIGNALS[LN1])
      else:
        SIGNALS[L0] = L0([SIGNALS[_] for _ in LN1] if (type(LN1)==list) else SIGNALS[LN1])

    self.infered_model = build_model(INPUTS, [SIGNALS[_] for _ in OUTPUTS], name=model.name)


class Darknet2Keras:

  def __init__(self,
    model_name,
    model_path,
    input_shape,
    input_name="image",
    output_dir=None
  ):
    self.input_shape  = input_shape
    self.input_name   = input_name
    self.model_name   = model_name
    self.model_path   = model_path
    self.output_dir   = output_dir
    os.makedirs(os.path.join(output_dir), exist_ok=True)

  @staticmethod
  def _load_config(config_file):
    section_ids = collections.defaultdict(int)
    ostream     = io.StringIO()

    for line in open(config_file).readlines():
      if line.startswith("["):          #  uniquify section names by appending a number id.
        section = line.strip().strip("[]")
        line    = line.replace(section, f"{section}_{section_ids[section]}")
        section_ids[section] += 1
      ostream.write(line)
    ostream.seek(0)

    config      = configparser.ConfigParser()
    config.read_file(ostream)
    return  config


  @staticmethod
  def _load_weight(weight_file):
    weight          = open(weight_file, "rb")
    major,minor,rev = np.ndarray((3), dtype="int32", buffer=weight.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
          seen = np.ndarray((1), dtype="int64", buffer=weight.read(8))
    else: seen = np.ndarray((1), dtype="int32", buffer=weight.read(4))
    weight_seek     = 12+seen.nbytes
    return  weight


  def convert(self,
    manipulate_fn=None,
    output_names=None,
  ):
    config_file     = self.model_path[0]
    weight_file     = self.model_path[1]
    config          = Darknet2Keras._load_config(config_file)
    weight          = Darknet2Keras._load_weight(weight_file)
    log_file        = f"{self.output_dir}/{self.model_name}.darknet2keras.log"
    log             = open(f"{self.output_dir}/{self.model_name}.darknet2keras.log","w")
    print(f"[I] Logs are written to '{log_file}'.")

    log.write(f"Load the DarkNet configuration in '{config_file}'.\n")
    log.write(f"Load the DarkNet weights in '{weight_file}'.\n\n")

    model_inputs    = keras.Input(self.input_shape, name=self.input_name)
    model_outputs   = []
    model_weights   = {}
    tensors_signal  = []
    x               = model_inputs
    weight_cnt      = 0

    for section in config.sections():
      log.write(f"Parse a section '{section}'\n")

      if   section.startswith("net"):
        try:    bn_epsilon = float(config[section]["batch_normalize_epsilon"])
        except: bn_epsilon = 1e-5
        continue
      elif section.startswith("region"):
        continue
      elif section.startswith("convolutional"):
        filters     = int(config[section]["filters"])
        size        = int(config[section]["size"   ])
        stride      = int(config[section]["stride" ])
        pad         = int(config[section]["pad"    ])
        activation  =     config[section]["activation"]
        batchnorm   = ("batch_normalize" in config[section])

        ##  Darknet weights formatted by [bias/beta, [gamma,mean,var], kernel].
        B           = np.ndarray(
            (filters),   dtype="float32", buffer=weight.read(filters*4)
        )
        weight_cnt += filters

        if batchnorm:
          _bn_weights = np.ndarray(     # gamma,____,mean,var
            (3,filters), dtype="float32", buffer=weight.read(filters*12)
          )                             # beta inserted below
          weight_cnt += 3*filters
          bn_weights  = np.insert(_bn_weights, 1, B, axis=0)

        channels    = x.shape[-1]
        K_size      = size*size*channels*filters
        K           = np.ndarray(
          (filters,channels,size,size), # NCHW format
          dtype="float32", buffer=weight.read(K_size*4)
        ).transpose((2,3,1,0))          # converted to HWCN format
        weight_cnt += K_size

        ##  Create Conv2D layer.
        if pad==1:
          if stride==2:
            P       = size//2           # expected number of paddings
            x       = ZeroPadding2D(padding=((P,P-1),(P,P-1)))(x)
            padding = "valid"
          else: padding = "same"
        else: padding = "valid"

        _op     = Conv2D(
          filters, (size,size),
          strides=stride,
          padding=padding,
          activation=None,
          use_bias=not(batchnorm),
        )
        x       = _op(x)
        model_weights[_op.name] = [K] if batchnorm else [K,B]

        if   batchnorm:
          _op   = BatchNormalization(epsilon=bn_epsilon)
          model_weights[_op.name] = bn_weights
          x     = _op(x)

        if   activation=="leaky"   :  x = LeakyReLU(0.1)(x)
        elif activation=="logistic":  x = Activation("sigmoid")(x)
        elif activation=="silu"    :  x = Activation("swish"  )(x)
        elif activation=="linear"  :  pass
        else: raise ValueError(f"Unknown activation='{activation}' provided.")
      elif section.startswith("route"):
        layers  = [tensors_signal[int(_)] for _ in config[section]["layers"].split(",")]

        if "groups" in config[section]:
          groups    = int(config[section]["groups"  ])
          group_id  = int(config[section]["group_id"])
          x         = tf.split(layers[0], groups, axis=-1)[group_id]
        else:
          x         = Concatenate()(layers) if len(layers)>1 else layers[0]
      elif section.startswith("maxpool"):
        size    = int(config[section]["size"  ])
        stride  = int(config[section]["stride"])
        x       = MaxPooling2D(size, stride, "same")(x)
      elif section.startswith("shortcut"):
        _from   = int(config[section]["from"])
        x       = Add()([tensors_signal[_from], x])
      elif section.startswith("upsample"):
        stride  = int(config[section]["stride"])
        x       = UpSampling2D(stride)(x)
      elif section.startswith("reorg"):
        stride  = int(config[section]["stride"])
        x       = tf.nn.space_to_depth(x, stride)
      elif section.startswith("yolo"):
        if not output_names: model_outputs.append(x)
      else: raise NotImplementedError(f"invalid section header '{section}'")

      tensors_signal.append(x)
      if output_names and x.name in output_names:
        model_outputs.append(x)
    ##  End of for(config.sections)

    ##  Create a Keras model.
    weight_remain = len(weight.read())//4
    assert  weight_remain==0, f"{weight_remain} unused weights"
    weight.close()
    log.write("\nAll weights consumed.")
    log.close()

    ##  Create a Keras model and set the weights.
    if len(model_outputs)==0: model_outputs.append(x)
    model_outputs.sort(key=lambda _: -_.shape[1]*_.shape[2])
    model         = keras.Model(model_inputs, model_outputs, name=self.model_name)

    for layer in model.layers:
      if isinstance(layer, (Conv2D, DepthwiseConv2D, Dense, BatchNormalization)):
        assert  layer.name in model_weights, f"No weights for '{layer.name}' provided"
        layer.set_weights(model_weights[layer.name])

    return  manipulate_fn(model) if manipulate_fn else model

  def __repr__(self):
    return  f"""{__class__.__name__}
    model_name    : {self.model_name}
    config_file   : {self.config_file}
    weight_file   : {self.weight_file}
    input_shape   : {str(self.input_shape).replace(" ","")}
    input_name    : {self.input_name}"""

  def __str__(self):  return self.__repr__()