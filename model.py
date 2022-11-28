import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers


IMG_SHAPE = (256, 256, 3)

class RevisitResNet50(tf.keras.Model):
  def __init__(self, name="revisit_resnet50V2", **kwargs):
    super(RevisitResNet50, self).__init__()
    self.backbone = ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    # avg. pooling
    self.avgpool_8 = layers.AveragePooling2D(pool_size=(1,1), strides=1, padding='valid', data_format=None)
    self.avgpool_4= layers.AveragePooling2D(pool_size=(2,2), strides=2, padding='valid', data_format=None)
    self.avgpool_2= layers.AveragePooling2D(pool_size=(4,4), strides=4, padding='valid', data_format=None)
    self.avgpool_1= layers.AveragePooling2D(pool_size=(8,8), strides=8, padding='valid', data_format=None)
    # 1x1 conv
    self.theta_8 = layers.Conv2D(1, (1,1), activation='sigmoid')
    self.theta_4 = layers.Conv2D(1, (1,1), activation='sigmoid')
    self.theta_2 = layers.Conv2D(1, (1,1), activation='sigmoid')
    self.theta_1 = layers.Conv2D(1, (1,1), activation='sigmoid')
    # dense layer
    self.fc = layers.Dense(1, activation='sigmoid')
    # loss
    self.loss_object = tf.keras.losses.BinaryCrossentropy()

  def call(self, input_img, label_8, label_4, label_2, label_1):
    F = self.backbone(input_img)
    # M8
    x = self.avgpool_8(F)
    M8 = self.theta_8(x)
    # M4
    x = self.avgpool_4(F)
    M4 = self.theta_4(x)
    # M2
    x = self.avgpool_2(F)
    M2 = self.theta_2(x)
    # M1
    x = self.avgpool_1(F)
    M1 = self.theta_1(x)
    # concatenate and predict
    x = layers.Concatenate(axis=1)([layers.Flatten()(M8), layers.Flatten()(M4), layers.Flatten()(M2), layers.Flatten()(M1)])
    y_pred = self.fc(x)
    # pyramid loss
    pyramid_loss = tf.reduce_mean([self.loss_object(label_8, M8), self.loss_object(label_4, M4), self.loss_object(label_2, M2), self.loss_object(label_1, M1)])
    self.add_loss(pyramid_loss)

    return y_pred
  
  def predict(self, img):
    F = self.backbone(img)
    # M8
    x = self.avgpool_8(F)
    M8 = self.theta_8(x)
    # M4
    x = self.avgpool_4(F)
    M4 = self.theta_4(x)
    # M2
    x = self.avgpool_2(F)
    M2 = self.theta_2(x)
    # M1
    x = self.avgpool_1(F)
    M1 = self.theta_1(x)
    # concatenate and predict
    x = layers.Concatenate(axis=1)([layers.Flatten()(M8), layers.Flatten()(M4), layers.Flatten()(M2), layers.Flatten()(M1)])
    y_pred = self.fc(x)
    return y_pred
  
  def build_model(self):
    x = layers.Input(shape=IMG_SHAPE)
    label_8 = layers.Input(shape=(8,8,1), dtype="float32")
    label_4 = layers.Input(shape=(4,4,1), dtype="float32")
    label_2 = layers.Input(shape=(2,2,1), dtype="float32")
    label_1 = layers.Input(shape=(1,1,1), dtype="float32")
    return Model(inputs=[x, label_8, label_4, label_2, label_1], outputs=self.call(x, label_8, label_4, label_2, label_1))