# -*- coding:utf-8 -*-

from keras.layers import Layer
import keras.backend as K

class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型，但是还是要用到训练好的转移矩阵
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)
    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        states = K.expand_dims(states[0], 2) # previous
        inputs = K.expand_dims(inputs, 2) # 这个时刻的对标签的打分值，Emission score
        trans = K.expand_dims(self.trans, 0) # 转移矩阵

        output = K.logsumexp(states+trans+inputs, 1) # e 指数求和，log是防止溢出
        return output, [output]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        # 在CRF中涉及到标签得分加上转移概率，而这个point score就是相当于是标签得分（在真是标签的情况下，查看预测对于真实标签位置的总得分），因为labels的shape是[B, T, N]，而在N这个维度是one-hot，
        # 这里再乘以pred，相当于是对labels存在1的地方进行打分，其余地方全为0，再进行第2个维度相加表示去除0的值，再相加表示求一个总的标签得分
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True) # 逐标签得分, shape [B, 1]
        labels1 = K.expand_dims(labels[:, :-1], 3) # shape [B, T-1, N, 1]
        labels2 = K.expand_dims(labels[:, 1:], 2) # shape [B, T-1, 1, N]
        # 这里相乘的目的相当于从上一时刻转移到当前时刻，确定当前时刻是从上一时刻哪一个标签转移过来的，因为labels是one-hot的形式，所以在最后两个维度只有1个元素为1，其他全部为0，表示转移标志
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分 shape [B, T-1, N, N]
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        # K.sum(trans*labels, [2, 3])，因为trans*labels的结果是[B, T-1, N, N], 而后面两个维度中只有1个有值，表示转移得分
        trans_score = K.sum(K.sum(trans*labels, [2, 3]), 1, keepdims=True) # 求出所有T-1时刻的概率转移总得分，K.sum(trans*labels, [2, 3]), 表示每个时刻的转移得分
        return point_score+trans_score # 两部分得分之和

    def call(self, inputs): # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred): # 目标y_pred需要是one hot形式
        mask = 1-y_true[:, 1:, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        init_states = [y_pred[:, 0]] # 初始状态
        log_norm, _, _ = K.rnn(self.log_norm_step, y_pred[:, 1:], init_states, mask=mask) # 计算Z向量（对数） shape[batch_size, output_dim]
        log_norm = K.logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）shape [batch_size, 1] 计算一个总的
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        return log_norm - path_score # 即log(分子/分母)

    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)