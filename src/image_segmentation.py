#%%
import tensorflow as tf
import tensorflow.keras as K
# !pip install -q git+https://github.com/tensorflow/examples.git
from tensorflow_examples.models.pix2pix import pix2pix
# !pip install tensorflow_datasets
import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
from IPython.display import clear_output
#%%
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#%%
import os
os.chdir(r'D:\LGVAE')
# os.chdir('/Users/anseunghwan/Documents/GitHub/LGVAE')
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
# data_directory = '/Users/anseunghwan/Documents/segmentation_datasets'
data_directory = 'D:/segmentation_datasets'
#%%
from subprocess import check_output

# folder_names = check_output(["ls", data_directory]).decode("utf8").split('\n')[:-1]
import re
folder_names = [x for x in os.listdir(data_directory) if not x.endswith('.txt')]

# images
img_list = []
for k in range(len(folder_names)):
    # file_names = check_output(["ls", data_directory + '/' + folder_names[k] + '/images']).decode("utf8").split('\n')[:-1]
    file_names = os.listdir(data_directory + '/' + folder_names[k] + '/images')
    for i in tqdm(range(len(file_names)), desc=folder_names[k]):
        img_list.append(plt.imread(data_directory + '/' + folder_names[k] + '/images/' + file_names[i]))
img_list = np.array(img_list)
print('images 데이터 크기: ', img_list.shape)

# normalization
img_list = (img_list - 127.5) / 127.5

# annotations
anno_list = []
for k in range(len(folder_names)):
    # file_names = check_output(["ls", data_directory + '/' + folder_names[k] + '/annotations/trimaps']).decode("utf8").split('\n')[:-1]
    file_names = os.listdir(data_directory + '/' + folder_names[k] + '/annotations/trimaps')
    for i in tqdm(range(len(file_names)), desc=folder_names[k]):
        anno_list.append(plt.imread(data_directory + '/' + folder_names[k] + '/annotations/trimaps/' + file_names[i])[:, :, [0]])
anno_list = np.array(anno_list)
print('annotation 데이터 크기: ', anno_list.shape)

code = np.unique(anno_list) # (object, background)
anno_list[np.where(anno_list == code[0])] = 1.0 # object
anno_list[np.where(anno_list == code[1])] = 0.0 # background

# plt.imshow(anno_list[0][:, :, 0], 'gray')

assert img_list.shape[0] == anno_list.shape[0]
#%%
from tensorflow.keras.utils import Sequence
import math

class Dataloader(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x = np.array([self.x[i] for i in indices])
        batch_y = np.array([self.y[i] for i in indices])
        
        batch_x = tf.cast(batch_x, tf.float32)
        batch_y = tf.cast(batch_y, tf.float32)
        
        if tf.random.uniform(()) > 0.50:
            batch_x = tf.image.flip_left_right(batch_x)
            batch_y = tf.image.flip_left_right(batch_y)

        return batch_x, batch_y

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
#%%
TRAIN_LENGTH = len(img_list)
EPOCHS = 20
BATCH_SIZE = 128
learning_rate = 0.0005
train_loader = Dataloader(img_list, anno_list, BATCH_SIZE, shuffle=True)
# a, b = next(iter(train_loader))
#%%
def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
#%%
# image, mask = next(iter(train_loader))
# sample_image, sample_mask = image[0], mask[0]
# display([sample_image, sample_mask])
#%%
OUTPUT_CHANNELS = 1 # binary classification
#%%
base_model = tf.keras.applications.MobileNetV2(input_shape=[32, 32, 3], include_top=False)
# base_model = tf.keras.applications.ResNet50(input_shape=[32, 32, 3], include_top=False)
# base_model.summary()

# use activation output layer for U-NET
layer_names = [
    'block_1_expand_relu',   
    'block_3_expand_relu',   
    'block_6_expand_relu',   
    'block_13_expand_relu',  
    'block_16_project',      
]
# layer_names = [
#     'conv1_relu',   
#     'conv2_block3_out',   
#     'conv3_block4_out',   
#     'conv4_block6_out',  
#     'conv5_block3_out',      
# ]
layers = [base_model.get_layer(name).output for name in layer_names]

# feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
# down_stack.summary()
down_stack.trainable = False
#%%
up_stack = [
    pix2pix.upsample(256, 4), 
    pix2pix.upsample(128, 4),  
    pix2pix.upsample(64, 4),  
    pix2pix.upsample(32, 4)
]
#%%
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[32, 32, 3])
    x = inputs

    # down sampling
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = K.layers.Concatenate()([x, skip])

    # last layer of U-NET
    last = K.layers.Conv2DTranspose(256, 4, strides=2,
                                    padding='same',
                                    activation='relu') 

    x = last(x)
    
    # output layer of U-NET
    last_output = K.layers.Conv2DTranspose(output_channels, 1, strides=1,
                                        padding='same',
                                        activation='linear') # linear logit

    x = last_output(x)

    return K.Model(inputs=inputs, outputs=x)
#%%
model = unet_model(OUTPUT_CHANNELS)
# model = unet()
model.summary()

model.compile(optimizer=K.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
#%%
# multi-class일 때만 사용
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]
#%%
def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image[tf.newaxis, ...])
            # display([image[0], mask[0], create_mask(pred_mask)])
            display([image, mask, pred_mask[0]])
    else:
        # display([sample_image, sample_mask,
        #         create_mask(model.predict(sample_image[tf.newaxis, ...]))])
        display([sample_image, sample_mask,
                model.predict(sample_image[tf.newaxis, ...])[0]])
#%%
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\n에포크 이후 예측 예시 {}\n'.format(epoch+1))
#%%
# VAL_SUBSPLITS = 5
# VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

# model_history = model.fit(train_loader, epochs=EPOCHS,
#                         #   validation_steps=VALIDATION_STEPS,
#                         #   validation_data=test_dataset,
#                           callbacks=[DisplayCallback()])

model_history = model.fit(train_loader, epochs=EPOCHS)
#%%
# loss = model_history.history['loss']
# # val_loss = model_history.history['val_loss']

# epochs = range(EPOCHS)

# plt.figure()
# plt.plot(epochs, loss, 'r', label='Training loss')
# # plt.plot(epochs, val_loss, 'bo', label='Validation loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss Value')
# plt.ylim([0, 1])
# plt.legend()
# plt.show()
#%%
model.save('./CIFAR10_assets/segmentation_model_{}_{}_{:.5f}.h5'.format(EPOCHS, BATCH_SIZE, learning_rate))
#%%
imported = K.models.load_model('./CIFAR10_assets/segmentation_model_{}_{}_{:.5f}.h5'.format(EPOCHS, BATCH_SIZE, learning_rate))
#%%
(x_train, y_train), (_, _) = K.datasets.cifar10.load_data()
'''-1 to 1 scaling'''
x_train = (x_train.astype('float32') - 127.5) / 127.5
#%%
'''evaluation'''
threshold = 0.2

import random
from copy import deepcopy
automobile = np.where(y_train == 1)[0]
random.seed(1)
idx = random.sample(range(len(automobile)), 50)
x_sample = x_train[automobile[idx]]
seg_sample = tf.nn.sigmoid(imported(x_sample))
x_seg = deepcopy(x_sample)
background = np.where(seg_sample < threshold)
x_seg[background[0], background[1], background[2], :] = 1.0
fig, axes = plt.subplots(10, 15, figsize=(15, 10))
for j in range(10):
    for k in range(5):
        axes[j][3*k].imshow((x_sample[10*k+j] + 1.0) / 2.0, 'gray')
        axes[j][3*k].axis('off')
        axes[j][3*k+1].imshow(seg_sample[10*k+j, :, :, 0])
        axes[j][3*k+1].axis('off')
        axes[j][3*k+2].imshow((x_seg[10*k+j] + 1.0) / 2.0)
        axes[j][3*k+2].axis('off')
plt.savefig('./CIFAR10_assets/segmentation_automobile_{}_{}_{:.5f}_{}.png'.format(EPOCHS, BATCH_SIZE, learning_rate, threshold),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%