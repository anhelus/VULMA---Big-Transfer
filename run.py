import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

from pdb import set_trace

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

SEEDS = 42

np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)

# train_ds, val_ds = tfds.load(
#     "tf_flowers",
#     split=["train[:85%]", "train[85%:]"],
#     as_supervised=True,
# )

# set_trace()

# TODO load dataset

# Define hyperparameters
RESIZE_TO = 224
CROP_TO = 48
BATCH_SIZE = 8
STEPS_PER_EPOCH = 32
AUTO = tf.data.AUTOTUNE # TODO check what it does
NUM_CLASSES = 7
SCHEDULE_LENGTH = (500) # train on lower resolution images
SCHEDULE_BOUNDARIES = [
    200,
    300,
    400
] 

# Preprocessiong helper functions

SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / BATCH_SIZE

train_ds = keras.utils.image_dataset_from_directory(
  'dataset',
  validation_split=0.2,
  subset="training",
  batch_size=None,
  seed=42)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'dataset',
  validation_split=0.2,
  subset="validation",
  batch_size=None,
  seed=42)

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.astype('uint8'))
    plt.title(int(label))
    plt.axis("off")

# set_trace()

plt.show()


@tf.function
def preprocess_train(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    # image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    image = image / 255.0
    return (image, label)

@tf.function
def preprocess_test(image, label):
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = image / 255.0
    return (image, label)

DATASET_NUM_TRAIN_EXAMPLES = train_ds.cardinality().numpy()

repeat_count = int(
    SCHEDULE_LENGTH * BATCH_SIZE / DATASET_NUM_TRAIN_EXAMPLES * STEPS_PER_EPOCH
)

repeat_count += 50 + 1 # almeno 50 epoche di training

# training pipeline
pipeline_train = (
    train_ds.shuffle(10000)
    .repeat(repeat_count)
    .map(preprocess_train, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

pipeline_validation = (
    val_ds.map(preprocess_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# image_batch, label_batch = next(iter(pipeline_train))

# plt.figure(figsize=(10, 10))
# for n in range(25):
#     ax = plt.subplot(5, 5, n + 1)
#     set_trace()
#     plt.imshow(image_batch[n])
#     plt.title(label_batch[n].numpy())
#     plt.axis("off")


# load pretrained tf-hub model into a keraslayer
bit_model_url = "https://tfhub.dev/google/bit/m-r50x1/1"

bit_module = hub.KerasLayer(bit_model_url)

class MyBiTModel(keras.Model):
    def __init__(self, num_classes, module, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.head = keras.layers.Dense(num_classes, kernel_initializer='zeros')
        self.bit_model = module
    
    def call(self, images):
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)


model = MyBiTModel(num_classes=NUM_CLASSES, module=bit_module)

# learning_rate = 0.003 * BATCH_SIZE / 512
learning_rate = 0.1

# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=SCHEDULE_BOUNDARIES,
    values=[
        learning_rate,
        learning_rate * 0.1,
        learning_rate * 0.01,
        learning_rate * 0.001,
    ],
)

# optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
optimizer = keras.optimizers.Adam()
# optimizer = keras.optimizers.SGD()

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

train_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    )
]

history = model.fit(
    pipeline_train,
    batch_size=BATCH_SIZE,
    epochs=int(SCHEDULE_LENGTH / STEPS_PER_EPOCH),
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=pipeline_validation,
    callbacks=train_callbacks,
)

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Training Progress")
    plt.ylabel("Accuracy/Loss")
    plt.xlabel("Epochs")
    plt.legend(["train_acc", "val_acc", "train_loss", "val_loss"], loc="upper left")
    plt.show()


plot_hist(history)

accuracy = model.evaluate(pipeline_validation)[1] * 100
print("Accuracy: {:.2f}%".format(accuracy))
