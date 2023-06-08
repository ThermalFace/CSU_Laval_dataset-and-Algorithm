import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import os
sys.path.append(r'./lib')
from ModeCore import CenterLoss, create_model_1

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATA_SIZE = [80, 64, 1]

import zipfile

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        print("Unzipping completed!")

# 压缩包目录
zip_path = './datasets/tufts.zip'
# 解压目录
extract_path = './datasets/tufts'

# 调用解压缩函数
unzip_file(zip_path, extract_path)

dataset_path = './datasets/tufts'
class_num = len(os.listdir(dataset_path))
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_path,
    image_size=DATA_SIZE[:-1],
    color_mode='grayscale',
    validation_split=0.2,
    subset="training",
    seed=100,
    batch_size=32
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_path,
    image_size=DATA_SIZE[:-1],
    color_mode='grayscale',
    validation_split=0.2,
    subset="validation",
    seed=100,
    batch_size=32
)
print(class_num)

model = create_model_1(input_shape=DATA_SIZE, output_len=class_num)

model.build(input_shape=[None]+DATA_SIZE)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
              loss=CenterLoss(num_classes=class_num),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)

history = model.fit(
    train_dataset,
    epochs=70,
    validation_data=val_dataset,
    steps_per_epoch=len(train_dataset),
    validation_steps=len(val_dataset)
)

# 保存参数
checkpoint.save(file_prefix = checkpoint_prefix)

# # 恢复参数
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

from matplotlib.pyplot import MultipleLocator
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
ax = plt.gca()
y_major_locator=MultipleLocator(0.05)
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(0.1, 1.05)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(['acc', 'val_acc'])
plt.show()
max_acc = tf.reduce_max(val_acc)
print(max_acc)


acc = history.history['loss']
val_acc = history.history['val_loss']
ax = plt.gca()
plt.plot(acc, label='Training Loss')
plt.plot(val_acc, label='Validation Loss')
plt.legend(['loss', 'val_loss'])
plt.show()
max_acc = tf.reduce_max(val_acc)
print(max_acc)
