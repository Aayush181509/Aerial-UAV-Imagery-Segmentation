import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.losses import BinaryCrossentropy
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import os
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import pickle
from keras.callbacks import ModelCheckpoint


seed=24
batch_size= 16
n_classes=2

scaler = MinMaxScaler()

#Use this to preprocess input for transfer learning
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

def preprocess_data(img, mask, num_class):
    #Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
    # Normalize mask values to 0 and 1
    mask = mask / 255
    #Convert mask to one-hot
    mask = to_categorical(mask, num_class)

    return (img,mask)

from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

def trainGenerator(train_img_path, train_mask_path, num_class):

    img_data_gen_args = dict(horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')

    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        batch_size = batch_size,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)

train_img_path = "Data/train_images/"
train_mask_path = "Data/train_masks/"
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=2)
val_img_path = "Data/val_images/"
val_mask_path = "Data/val_masks/"
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=2)


x, y = train_img_gen.__next__()
x_val, y_val = val_img_gen.__next__()

num_train_imgs = len(os.listdir('Data/train_images/train/'))
num_val_images = len(os.listdir('Data/val_images/val/'))
steps_per_epoch = num_train_imgs//batch_size
val_steps_per_epoch = num_val_images//batch_size

IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]
n_classes=2

# Define the model
model = sm.Unet(BACKBONE, encoder_weights='imagenet',
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                classes=n_classes, activation='sigmoid')
model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=[sm.metrics.iou_score], run_eagerly=True)

class_weights = {0: 1, 1: 7}



# Define a ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    filepath='solidwaste_{epoch:02d}.hdf5',  # File path to save the model
    save_best_only=True,
    monitor='val_loss',   # Save the model at every epoch
    save_freq='epoch'  # Save the model after every epoch
)


history=model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch,
          class_weight=class_weights,
          callbacks=[model_checkpoint])

model.save('Final_UNET_RESNET_backbone.hdf5')
# havent include optimizer state. checkpoint

# Save the training history to a file

with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)