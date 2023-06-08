import tensorflow as tf
from Attention import SelfAttention, ChannelAttention


def get_Sub_model(input_shape=[64, 80, 1], output_len=10):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(scale=1./255)(inputs)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x_1 = tf.keras.layers.Conv2D(128, (1, 1), strides=1, padding='same')(x)
    x_2 = tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x_3 = tf.keras.layers.Conv2D(128, (5, 5), strides=1, padding='same')(x)

    x = tf.concat([x_1, x_2, x_3], axis=-1)

    x = SelfAttention()(x)
    x = ChannelAttention()(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    logits = tf.keras.layers.Dense(output_len)(x)
    outputs = tf.keras.layers.Softmax()(logits)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model



# 定义模型
def create_model_1(input_shape=(32, 32, 1), output_len=10):
    model = get_Sub_model(input_shape=input_shape, output_len=output_len)
    
    return model

# 定义模型
def create_model_1_AMsoftmax(input_shape=(32, 32, 1), output_len=10):
    base_model = get_Sub_model(input_shape=input_shape, output_len=output_len)
    amsoftmax_output = Amsoftmax(num_classes=output_len)(base_model.layers[-2].output)
    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=amsoftmax_output)
    
    return model
    


import tensorflow as tf

class CenterLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        
        # 初始化中心点，形状为 (num_classes, feature_dim)
        self.centers = tf.Variable(initial_value=tf.zeros((num_classes,)), trainable=False)
        
    def call(self, y_true, y_pred):
        # y_true: ground truth labels, shape=(batch_size,)
        # y_pred: predicted logits, shape=(batch_size, num_classes)
        
        # 计算每个样本对应的中心点
        centers_batch = tf.gather(self.centers, y_true)
        
        # 计算每个样本与其对应中心点的距离
        diff = y_pred - centers_batch
        
        # 计算center loss
        loss = tf.reduce_mean(tf.square(diff))

        # 交叉熵损失
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        return self.alpha * loss + (1-self.alpha) * cross_entropy_loss
    
    def update_centers(self, features, labels):
        # features: shape=(batch_size, feature_dim)
        # labels: shape=(batch_size,)
        
        # 按标签计算每个类别的样本均值
        centers_batch = tf.TensorArray(tf.float32, size=self.num_classes)
        for c in tf.range(self.num_classes):
            mask = tf.equal(labels, c)
            mask = tf.reshape(mask, [-1])
            feat = tf.boolean_mask(features, mask)
            centers_batch = centers_batch.write(c, tf.reduce_mean(feat, axis=0))
        
        # 更新中心点
        centers_batch = centers_batch.stack()
        self.centers.assign(centers_batch)


  

import tensorflow as tf

class Amsoftmax(tf.keras.layers.Layer):
    def __init__(self, num_classes, scale=30, **kwargs):
        super(Amsoftmax, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.num_classes),
                                      initializer='glorot_uniform',
                                      trainable=True)
        
    def call(self, inputs, **kwargs):
        # normalize the input vector
        inputs = tf.math.l2_normalize(inputs, axis=1)
        
        # normalize the weights
        kernel = tf.math.l2_normalize(self.kernel, axis=0)
        
        # calculate the logits
        logits = tf.matmul(inputs, kernel)
        
        # add the scale factor to the logits
        logits = self.scale * logits
        
        # apply the softmax function
        probabilities = tf.nn.softmax(logits)
        
        return probabilities
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)




