from model import RevisitResNet50
from datetime import datetime
import tensorflow as tf
from load_dataset import DataLoader1
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
save_model_path = os.path.join(ROOT_DIR, 'saved_models/cp.ckpt')
data_path = os.path.join(ROOT_DIR, 'rgb.zip')
train = DataLoader1(data_path, batch_size=16)


optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, decay=5e-5)
bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
loss_metric = tf.keras.metrics.Mean()


revisit_model = RevisitResNet50()
revisit_model.build_model()

epochs = 30
start = datetime.now()
for epoch in range(0,1):
  print("Start of epoch %d" % (epoch,))

  # Iterate over the batches of the dataset
  for step, batch_train in enumerate(train.dataset):
    # print(batch_train['image'].shape, batch_train['label'].shape, batch_train['label_8'].shape, batch_train['label_4'].shape, batch_train['label_2'].shape, batch_train['label_1'].shape)
    with tf.GradientTape() as tape:
        pred = revisit_model(batch_train['image'], batch_train['label_8'], batch_train['label_4'], batch_train['label_2'], batch_train['label_1'], training=True)
        # compute bcse loss
        loss = bce_loss_fn(batch_train['label'], pred)
        loss += sum(revisit_model.losses)  # add pyramid loss

    grads = tape.gradient(loss, revisit_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, revisit_model.trainable_weights))

    loss_metric(loss)
    print(step)
    if (step+1) % 50 == 0:
      diff_time = datetime.now()-start
      days, seconds = diff_time.days, diff_time.seconds
      hours = days * 24 + seconds // 3600
      minutes = (seconds % 3600) // 60
      seconds = seconds % 60
      print("step %d: mean loss = %.6f, time = %d:%d:%d" % (step+1, loss_metric.result(), hours, minutes, seconds))
      revisit_model.save_weights(save_model_path)

revisit_model.save_weights(save_model_path)
