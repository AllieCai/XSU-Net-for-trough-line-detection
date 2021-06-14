import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
# from keras.optimizers import SGD
# from sklearn.metrics import log_loss
from keras.applications.inception_v3 import InceptionV3

# def SE_model(nb_classes, channel,input_shape=(240,400, 1)):
#     inputs_dim = Input(input_shape)
#     x = InceptionV3(include_top=False,
#                 weights='imagenet',
#                 input_tensor=None,
#                 input_shape=(240,400, 1),
#                 pooling=max)(inputs_dim)
#     print(x.shape)
#     # max_pooling = MaxPooling2D(10,10)(x)
#     squeeze = GlobalAveragePooling2D()(x)
#
#     excitation = Dense(units=channel // 16)(squeeze)
#     excitation = Activation('relu')(excitation)
#     excitation = Dense(units=channel)(excitation)
#     excitation = Activation('sigmoid')(excitation)
#     excitation = Reshape((1, 1, channel))(excitation)
#
#
#     scale = multiply([x, excitation])
#
#     x = GlobalAveragePooling2D()(scale)
#     dp_1 = Dropout(0.3)(x)
#     fc2 = Dense(nb_classes)(dp_1)
#     fc2 = Activation('sigmoid')(fc2) #此处注意，为sigmoid函数
#     model = Model(inputs=inputs_dim, outputs=fc2)
#     return model


def model(pretrained_weights=None, input_size=(240,400, 1)):
	inputs = Input(input_size)

	# Block 1
	x1 = Conv2D(32, 3, strides=(2, 2), use_bias=False)(inputs)
	x1 = BatchNormalization()(x1)
	x1 = Activation('relu')(x1)
	x1 = Conv2D(64, 3, use_bias=False)(x1)
	x1 = BatchNormalization()(x1)
	x1= Activation('relu')(x1)

	residual = Conv2D(128, 1, strides=(2, 2), padding='same', use_bias=False)(x1)
	residual = BatchNormalization()(residual)

	# Block 2
	x2 = SeparableConv2D(128, 3, padding='same', use_bias=False)(x1)
	x2 = BatchNormalization()(x2)
	x2 = Activation('relu')(x2)
	x2 = SeparableConv2D(128, 3, padding='same', use_bias=False)(x2)
	x2 = BatchNormalization()(x2)

	# Block 2 Pool
	x2 = MaxPooling2D(3, strides=(2, 2), padding='same')(x2)
	x2 = layers.add([x1, residual])

	residual = Conv2D(256, 1, strides=(2, 2), padding='same', use_bias=False)(x2)
	residual = BatchNormalization()(residual)

	# Block 3
	x3 = Activation('relu')(x2)
	x3 = SeparableConv2D(256, 3, padding='same', use_bias=False)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation('relu')(x3)
	x3 = SeparableConv2D(256, 3, padding='same', use_bias=False)(x3)
	x3 = BatchNormalization()(x3)

	# Block 3 Pool
	x3 = MaxPooling2D(3, strides=(2, 2), padding='same')(x3)
	x3 = layers.add([x2, residual])

	residual = Conv2D(512, 1, strides=(2, 2), padding='same', use_bias=False)(x3)
	residual = BatchNormalization()(residual)

	# Block 4
	x4 = Activation('relu')(x3)
	x4 = SeparableConv2D(512, 3, padding='same', use_bias=False)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)
	x4 = SeparableConv2D(512, 3, padding='same', use_bias=False)(x4)
	x4 = BatchNormalization()(x4)

	x4 = MaxPooling2D(3, strides=(2, 2), padding='same')(x4)
	x4 = layers.add([x3, residual])

	# Block 5 - 12
	for i in range(8):
		residual = x4

		x4 = Activation('relu')(x4)
		x4 = SeparableConv2D(512, 3, padding='same', use_bias=False)(x4)
		x4 = BatchNormalization()(x4)
		x4 = Activation('relu')(x4)
		x4 = SeparableConv2D(512, 3, padding='same', use_bias=False)(x4)
		x4 = BatchNormalization()(x4)
		x4 = Activation('relu')(x4)
		x4 = SeparableConv2D(512, 3, padding='same', use_bias=False)(x4)
		x4 = BatchNormalization()(x4)

		x5 = layers.add([x4, residual])

	residual = Conv2D(1024, 1, strides=(2, 2), padding='same', use_bias=False)(x5)
	residual = BatchNormalization()(residual)

	# Block 13
	x5 = Activation('relu')(x5)
	x5 = SeparableConv2D(1024, 3, padding='same', use_bias=False)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation('relu')(x5)
	x5 = SeparableConv2D(1024, 3, padding='same', use_bias=False)(x5)
	x5 = BatchNormalization()(x5)

	# Block 13 Pool
	x5 = MaxPooling2D(3, strides=(2, 2), padding='same')(x5)
	x6 = layers.add([x5, residual])

	# Block 14
	x6 = SeparableConv2D(1024, 3, padding='same', use_bias=False)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation('relu')(x6)

	# Block 14 part 2
	x6 = SeparableConv2D(1024, 3, padding='same', use_bias=False)(x6)
	x6 = BatchNormalization()(x6)
	x7 = Activation('relu')(x6)


	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(x7))
	merge6 = concatenate([x4, up6], axis=3)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
	squeeze = GlobalAveragePooling2D()(conv6)

	excitation = Dense(units=512 // 16)(squeeze)
	excitation = Activation('relu')(excitation)
	excitation = Dense(units=512)(excitation)
	excitation = Activation('sigmoid')(excitation)
	excitation = Reshape((1, 1, 512))(excitation)

	scale = multiply([conv6, excitation])

	x = GlobalAveragePooling2D()(scale)
	dp_1 = Dropout(0.3)(x)
	fc2 = Dense(2)(dp_1)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(fc2))
	merge7 = concatenate([x3, up7], axis=3)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
	squeeze = GlobalAveragePooling2D()(conv7)

	excitation = Dense(units=256 // 16)(squeeze)
	excitation = Activation('relu')(excitation)
	excitation = Dense(units=256)(excitation)
	excitation = Activation('sigmoid')(excitation)
	excitation = Reshape((1, 1, 56))(excitation)

	scale = multiply([conv7, excitation])

	x = GlobalAveragePooling2D()(scale)
	dp_2 = Dropout(0.3)(x)
	fc3 = Dense(2)(dp_2)
	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(fc3))
	merge8 = concatenate([x2, up8], axis=3)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
	squeeze = GlobalAveragePooling2D()(conv8)

	excitation = Dense(units=128 // 16)(squeeze)
	excitation = Activation('relu')(excitation)
	excitation = Dense(units=128)(excitation)
	excitation = Activation('sigmoid')(excitation)
	excitation = Reshape((1, 1, 56))(excitation)

	scale = multiply([conv8, excitation])

	x = GlobalAveragePooling2D()(scale)
	dp_3 = Dropout(0.3)(x)
	fc4 = Dense(2)(dp_3)
	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(fc4))
	merge9 = concatenate([x1, up9], axis=3)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	squeeze = GlobalAveragePooling2D()(conv7)

	excitation = Dense(units=64 // 4)(squeeze)
	excitation = Activation('relu')(excitation)
	excitation = Dense(units=64)(excitation)
	excitation = Activation('sigmoid')(excitation)
	excitation = Reshape((1, 1, 56))(excitation)

	scale = multiply([conv9, excitation])

	x = GlobalAveragePooling2D()(scale)
	dp_3 = Dropout(0.3)(x)
	fc4 = Dense(2)(dp_3)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(fc4)
	conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

	model = Model(input=inputs, output=conv10)

	model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

	# model.summary()
	if (pretrained_weights):
		model.load_weights(pretrained_weights)


	return model



