from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keras import backend as K

# 导入VGG-16模型和权重
model = VGG16(weights='imagenet')
# 查看模型结构
model.summary()
# 导入图片
src = cv.imread("./0.png")
# 改变图片尺寸适应网络结构
img = cv.resize(src, (224, 224))
# 添加一个维度
x = np.expand_dims(img, 0)

# 模型预测
preds = model.predict(x)
# 打印出top3预测
print('Predicted:', decode_predictions(preds, top=3)[0])
# 打印索引
print(np.argmax(preds[0]))

# 这是预测向量中的“非洲象”条目
african_elephant_output = model.output[:, np.argmax(preds[0])]
# 获得VGG-16中的最后一个卷积层
last_conv_layer = model.get_layer('block5_conv3')
# 求“非洲象”类相对于“block5_conv3”输出特征映射的梯度
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
# 改层是一个形状向量(512，)，其中每个条目是特定feature map通道上梯度的平均强度
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# 给定一个样本图像，获取输出
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
# 这些是这两个量的值，作为Numpy数组，给出了两个大象的样本图像
pooled_grads_value, conv_layer_output_value = iterate([x])

# 将feature map数组中的每个通道乘以关于elephant类的“这个通道有多重要”
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
# 得到的特征图的通道平均是我们的类激活热图
heatmap = np.mean(conv_layer_output_value, axis=-1)
# 归一化处理
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# 数组绘制成图像
plt.matshow(heatmap)
plt.show()

# 调整热图的大小，使其与原始图像大小相同
heatmap = cv.resize(heatmap, (src.shape[1], src.shape[0]))
# 将热图转换为RGB
heatmap = np.uint8(255 * heatmap)
# 转化为伪彩色图像
heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
# 进行叠加
superimposed_img = heatmap * 0.5 + src
# 将预测结果添加
cv.putText(superimposed_img, decode_predictions(preds)[0][0][1],(50, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2, 8)
# 保存叠加图像
cv.imwrite('./elephant.png', superimposed_img)