from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from pylab import *
import keras
import cv2


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    feature_map_combination = []
    plt.figure()
    num_pic = feature_map.shape[2]  # 获取通道数（featuremap数量）
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
        title('feature_map_{}'.format(i))

    plt.savefig('feature_map.tif')
    plt.show()

    # 各个特征图按1:1
    feature_map_sum = sum(ele for ele in feature_map_combination)

    feature_map_sum = (feature_map_sum - np.min(feature_map_sum)) / (
                np.max(feature_map_sum) - np.min(feature_map_sum))  # 融合后进一步归一化
    y_predict = np.array(feature_map_sum).astype('float')
    y_predict = np.round(y_predict, 0).astype('uint8')
    y_predict *= 255
    y_predict = np.squeeze(y_predict).astype('uint8')
    cv2.imwrite("C:\\Users\\charl\\0.tif", y_predict)

    plt.imshow(y_predict)
    plt.savefig("y_predict.tif")


# def create_model():  # 创建模型方法一


# model = Sequential()
#    model.add(Convolution2D(9, 1, 1, input_shape=img.shape))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(4, 4)))
def unet():  # 创建模型方法一
    inputs = Input(img.shape)
    # 第一组两个卷积
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # # 第二组两个卷积
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # #第三组两个卷积
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #第四组两个卷积
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    #最底层两个卷积
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # 第一个上采样 和concate
    # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    # merge6 = concatenate([drop4,up6], axis = 3)
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #    #第二个上采样 和concate
    # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    # merge7 = concatenate([conv3,up7], axis = 3)
    # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    # #第三个上采样 和concate
    # up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    # merge8 = concatenate([conv2,up8], axis = 3)
    # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    # #第四个上采样 和concate
    # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    # merge9 = concatenate([conv1,up9], axis = 3)
    # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv1 = Conv2D(1, 3, activation='sigmoid')(conv3)
    model = Model(input=inputs, output=conv1)
    opt = Adam(lr=1e-4)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    img = cv2.imread('1.jpg')
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # 输入归一化
    img_batch = np.expand_dims(img, axis=0)
    print("ss=", img.shape)
    model = unet()
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
    print(img_batch.shape)
    print(conv_img.shape)
    print("conv_img=", conv_img)
    print(conv_img.shape)
    visualize_feature_map(conv_img)
