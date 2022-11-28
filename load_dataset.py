import numpy as np
import zipfile, cv2, random
import tensorflow as tf

def new_py_function(func, inp, Tout, name=None):
  def wrapped_func(*flat_inp):
    reconstructed_inp = tf.nest.pack_sequence_as(inp, flat_inp,expand_composites=True)
    out = func(*reconstructed_inp)
    return tf.nest.flatten(out, expand_composites=True)
  flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
  flat_out = tf.py_function(
      func=wrapped_func, 
      inp=tf.nest.flatten(inp, expand_composites=True),
      Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
      name=name)
  spec_out = tf.nest.map_structure(_dtype_to_tensor_spec, Tout, 
                                   expand_composites=True)
  out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
  return out

def _dtype_to_tensor_spec(v):
  return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v

def _tensor_spec_to_dtype(v):
  return v.dtype if isinstance(v, tf.TensorSpec) else v

class DataLoader1:
  def __init__(self, dataset_path, batch_size=4, image_size=(256, 256), shuffle=True):
    self.archive = zipfile.ZipFile(dataset_path, 'r')
    self.images_paths = [k for k in list(self.archive.namelist()) if '.png' in k]
    self.images_paths1 = [k for k in list(self.archive.namelist()) if '1_' in k]
    print("Zalo Dataset create sucsess. Total: {}".format(len(self.images_paths)))
    print("Live: {}".format(len(self.images_paths1)))
    print("Spoof: {}".format(len(self.images_paths)-len(self.images_paths1)))
    
    random.shuffle(self.images_paths)
    self.dim = image_size
    dataset = tf.data.Dataset.from_tensor_slices(self.images_paths).map(
            self.pad_map_fn, num_parallel_calls=3
          )
    if shuffle:
      dtaset = dataset.shuffle(buffer_size=100000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(buffer_size=3)
    # dataset = dataset.repeat()
    self.dataset = dataset.apply(tf.data.experimental.ignore_errors())  
  def __len__(self):
        return self.length

  def load_tf_image(self, image_path):
    image_path = image_path.numpy().decode("utf-8")
    img_data = self.archive.read(image_path.strip())
    image = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, self.dim)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # label - spoof = 0, real = 1
    if '0_' in image_path:
      label =  tf.constant([0])
      label_8 = tf.zeros_like(tf.random.uniform(shape=(8,8,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
      label_4 = tf.zeros_like(tf.random.uniform(shape=(4,4,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
      label_2 = tf.zeros_like(tf.random.uniform(shape=(2,2,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
      label_1 = tf.zeros_like(tf.random.uniform(shape=(1,1,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
    if '1_' in image_path:
      label = tf.constant([1])
      label_8 = tf.ones_like(tf.random.uniform(shape=(8,8,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
      label_4 = tf.ones_like(tf.random.uniform(shape=(4,4,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
      label_2 = tf.ones_like(tf.random.uniform(shape=(2,2,1), minval=0, maxval=1, dtype=tf.dtypes.float32))
      label_1 = tf.ones_like(tf.random.uniform(shape=(1,1,1), minval=0, maxval=1, dtype=tf.dtypes.float32))

    return {'image': image, 'label': label, 'label_8': label_8, 'label_4': label_4, 'label_2': label_2, 'label_1': label_1}

  def pad_map_fn(self, img_path):
    return new_py_function(self.load_tf_image, inp=[img_path], Tout=({"image": tf.float32, "label": tf.int32, 'label_8': tf.float32, 'label_4': tf.float32, 'label_2': tf.float32, 'label_1': tf.float32}))


