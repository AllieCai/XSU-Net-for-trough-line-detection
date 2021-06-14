from model import *
from data import *
from keras.callbacks import TensorBoard
import os
import matplotlib as plt
import keras

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(1,'data1/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = model()

model.compile(Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

TensorBoard(log_dir='./log/',
                    histogram_freq= 0 ,
                    write_graph=True,
                write_images=True)
tbCallBack = TensorBoard(log_dir='./log/',
                                         histogram_freq= 0,
                                         write_graph=True,
                                         write_images=True)

hist = model.fit_generator(myGene,steps_per_epoch=7000,epochs=150,callbacks=[model_checkpoint])
# training_vis(hist)100
testGene = testGenerator("data1/membrane/test")
results = model.predict_generator(testGene,680,verbose=1)
saveResult("data1/membrane/test",results)