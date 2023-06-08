import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, gamma=0.01, trainable=True):
        super().__init__(trainable=trainable)
        self._gamma = gamma
        self.gamma = gamma
        self.f = None
        self.g = None
        self.h = None
        self.v = None
        self.attention = None

    def build(self, input_shape):
        c = input_shape[-1]
        self.f = self.block(c//8)     # reduce channel size, reduce computation
        self.g = self.block(c//8)     # reduce channel size, reduce computation
        self.h = self.block(c//8)     # reduce channel size, reduce computation
        self.v = tf.keras.layers.Conv2D(c, 1, 1)              # scale back to original channel size
        # global GAMMA_id
        # self.gamma = self.add_weight(
        #     "gamma{}".format(GAMMA_id), shape=None, initializer=tf.keras.initializers.constant(self._gamma))
        # GAMMA_id += 1

    @staticmethod
    def block(c):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(c, 1, 1),   # [n, w, h, c] 1*1conv
            tf.keras.layers.Reshape((-1, c)),          # [n, w*h, c]
        ])

    def call(self, inputs, **kwargs):
        f = self.f(inputs)    # [n, w, h, c] -> [n, w*h, c//8]
        g = self.g(inputs)    # [n, w, h, c] -> [n, w*h, c//8]
        h = self.h(inputs)    # [n, w, h, c] -> [n, w*h, c//8]
        s = tf.matmul(f, g, transpose_b=True)   # [n, w*h, c//8] @ [n, c//8, w*h] = [n, w*h, w*h]
        self.attention = tf.nn.softmax(s, axis=-1)
        context_wh = tf.matmul(self.attention, h)  # [n, w*h, w*h] @ [n, w*h, c//8] = [n, w*h, c//8]
        s = inputs.shape        # [n, w, h, c]
        cs = context_wh.shape   # [n, w*h, c//8]
        context = tf.reshape(context_wh, [-1, s[1], s[2], cs[-1]])    # [n, w, h, c//8]
        o = self.v(self.gamma * context) + inputs   # residual
        return o
    

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.reduction_ratio = reduction_ratio
        
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units=self.reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=self.reduction_ratio, activation='relu')
        self.fc3 = tf.keras.layers.Dense(units=1, activation='sigmoid')
        
    def call(self, inputs):
        # Compute the average pooled feature map
        avg_pooled = self.global_avg_pool(inputs)
        
        # Pass it through two fully connected layers
        fc1_output = self.fc1(avg_pooled)
        fc2_output = self.fc2(fc1_output)
        
        # Apply sigmoid activation and reshape for broadcasting
        sigmoid_output = self.fc3(fc2_output)
        sigmoid_output = tf.reshape(sigmoid_output, shape=(-1, 1, 1, sigmoid_output.shape[-1]))
        
        # Multiply the original feature map with the attention weights
        attention_weights = inputs * sigmoid_output
        output = attention_weights + inputs
        
        return output
