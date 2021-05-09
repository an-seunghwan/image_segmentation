#%%
'''
<reference>
re-implementation code of https://github.com/zhixuhao/unet

- only binary classification
'''
#%%
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras
#%%
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte

os.chdir('/home/jeon/Desktop/an/image_segmentation')
#%%
'''image input normalization'''
def normalize(img, mask):
    img = img / 255.0
    mask = mask / 255.0
    mask[mask > 0.5] = 1.0
    mask[mask <= 0.5] = 0.0
    return (img, mask)
#%%
'''augmentation'''
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
#%%
# batch_size = 20
# train_path = 'data/membrane/train'
# image_folder = 'image'
# mask_folder = 'label'
# aug_dict = data_gen_args
# image_color_mode = "grayscale"
# mask_color_mode = "grayscale"
# image_save_prefix = "image"
# mask_save_prefix  = "mask"
# save_to_dir = "data/membrane/train/aug"
# target_size = (256, 256)
# seed = 1
#%%
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)

# image_generator = image_datagen.flow_from_directory(
#     train_path,
#     classes = [image_folder],
#     class_mode = None,
#     color_mode = image_color_mode,
#     target_size = target_size,
#     batch_size = batch_size,
#     save_to_dir = save_to_dir,
#     save_prefix  = image_save_prefix,
#     seed = seed)

# mask_generator = mask_datagen.flow_from_directory(
#     train_path,
#     classes = [mask_folder],
#     class_mode = None,
#     color_mode = mask_color_mode,
#     target_size = target_size,
#     batch_size = batch_size,
#     save_to_dir = save_to_dir,
#     save_prefix  = mask_save_prefix,
#     seed = seed)

# train_generator = zip(image_generator, mask_generator)

# augmentation이 수행될 때마다 image와 mask를 저장
# sampleimg, samplemask = next(iter(train_generator))
# print(sampleimg.shape)
# print(samplemask.shape)
#%%
def BuildTrainGenerator(batch_size,
                    train_path,
                    image_folder,
                    mask_folder,
                    aug_dict,
                    image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",
                    image_save_prefix  = "image",
                    mask_save_prefix  = "mask",
                    save_to_dir = None,
                    target_size = (256, 256), # reshape image 512x512 -> 256x256
                    seed = 1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = normalize(img, mask)
        yield (img, mask) # generate image on demand
#%%
traingenerator = BuildTrainGenerator(10, 
                            'data/membrane/train', 
                            'image',
                            'label', 
                            data_gen_args) 
#%%
'''
model architecture
(# of parameters: 31,031,685)
'''
def BuildUnet(input_size = (256, 256, 1)):
    
    '''contracting path'''
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1) # 256x256x64
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 128x128x64
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2) # 128x128x128
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 64x64x128
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3) # 64x64x256
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 32x32x256
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) # 32x32x512
    drop4 = Dropout(0.5)(conv4) # 32x32x512, implicit augmentation
    
    '''bottle-neck'''
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # 16x16x512
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5) # 16x16x1024, implicit augmentation

    '''expanding path'''
    updrop5 = UpSampling2D(size = (2, 2))(drop5) # 32x32x1024
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(updrop5) # 32x32x512
    merge6 = concatenate([drop4, up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    upconv6 = UpSampling2D(size = (2, 2))(conv6) # 64x64x512
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upconv6) #64x64x256
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    upconv7 = UpSampling2D(size = (2, 2))(conv7) # 128x128x256
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upconv7) # 128x128x128
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    upconv8 = UpSampling2D(size = (2, 2))(conv8) # 256x256x128
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upconv8) # 256x256x64
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) # 256x256x2, final feature map
    
    '''output layer'''
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    return model
#%%
'''train'''
model = BuildUnet()
# model_checkpoint = ModelCheckpoint('./assets/unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
# model.fit(traingenerator, steps_per_epoch=10, epochs=1, callbacks=[model_checkpoint])
model.fit(traingenerator, steps_per_epoch=4000, epochs=5) # no callbacks
# last accuracy: 0.9791
#%%
model.save_weights('./assets/weights')
#%%
imported = BuildUnet()
imported.load_weights('./assets/weights').expect_partial()
#%%
def BuildTestGenerator(test_path,
                        num_image = 30,
                        target_size = (256, 256),
                        as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "{}.png".format(i)), as_gray = as_gray)
        img = img / 255.0
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,)+img.shape)
        yield img
#%%
def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, "predict_{}.png".format(i)), img_as_ubyte(img))
#%%
'''test result'''
testgenerator = BuildTestGenerator("data/membrane/test")
results = imported.predict(testgenerator, 30, verbose=1)
saveResult("data/membrane/test", results)
#%%
# '''test set result'''
# import matplotlib.pyplot as plt
# testgenerator = BuildTestGenerator("data/membrane/test")

# for i, testimg in enumerate(testgenerator):
#     fig, axes = plt.subplots(1, 2, figsize=(6, 3))
#     axes.flatten()[0].imshow(trans.resize(testimg[0, ...], (256, 256, 1)), 'gray')
#     axes.flatten()[1].imshow(results[i, ...], 'gray')
#     plt.tight_layout()
#     plt.show()
#     plt.close()
#     if (i >= 4): break
#%%