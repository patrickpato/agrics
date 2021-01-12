import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras import layers
from tensorflow.keras import callbacks



DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argcs[1] == '--development' or argvs[1] == '-d'):
    DEV = True
if DEV:
    epochs=2 
else:
    epochs=150

train_data_path = '/home/dev-works/Desktop/tensorflow/disease_classification/data/train'
val_data_path = '/home/dev-works/Desktop/tensorflow/disease_classification/data/val'

img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 5
validation_steps = 10
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size =2
classes_num = 3
lr = 1e-4

model = Sequential()
model.add(layers.Conv2D(nb_filters1, 3,3, padding = 'same', input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(layers.Conv2D(nb_filters2, 2,2, padding = 'same'))
model.add(Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(128, 2,2, padding='same'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer=optimizers.RMSprop(lr=lr), metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1. / 255, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale=1. /255)

train_generator = train_datagen.flow_from_directory(train_data_path, 
        target_size = (img_height, img_width), 
        batch_size = batch_size, 
        class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory(val_data_path, 
        target_size = (img_height, img_width), 
        batch_size = batch_size, 
        class_mode = 'categorical')
log_dir = '.tf-log/'
tb_cv = callbacks.TensorBoard(log_dir=log_dir, histogram_freq = 0)
cbks = [tb_cv]


model.fit_generator(train_generator,  
        epochs = epochs, 
        validation_data = validation_generator, 
        callbacks = cbks, 
        validation_steps = validation_steps)

target_dir = '/home/dev-works/Desktop/tensorflow/disease_classification/models/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save('/home/dev-works/Desktop/tensorflow/disease_classification/models/model.h5')
model.save_weights('/home/dev-works/Desktop/tensorflow/disease_classification/models/weights.h5')

