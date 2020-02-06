import tensorflow as tf
from tensorflow.keras.layers import Dense, \
    Conv2D, Flatten, BatchNormalization, LeakyReLU, Input
from tensorflow.keras import Model


class DeepQNetArch(tf.keras.Model):

    def __init__(self, n_outputs, input_shape, trainable=True, name='dqn_architecture'):
        super().__init__(name=name)

        self._input_shape = input_shape

        self.input_layer = Conv2D(32, (8, 8), strides=(4, 4), input_shape=input_shape,
                                  padding='valid', kernel_initializer='he_normal',
                                  name=name + '_conv1', trainable=trainable)
        self.lrelu1 = LeakyReLU(name=name + '_input_layer')

        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2),
                            padding='valid', kernel_initializer='he_normal',
                            name=name + '_conv2', trainable=trainable)
        self.lrelu2 = LeakyReLU(name=name + '_lrelu2')

        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1),
                            padding='valid', kernel_initializer='he_normal',
                            name=name + '_conv3', trainable=trainable)
        self.lrelu3 = LeakyReLU(name=name + '_lrelu3')

        self.flatten = Flatten(name=name + '_flatten')

        self.dense1 = Dense(512, kernel_initializer='he_normal', name=name + '_dense1', trainable=trainable)
        self.lrelu4 = LeakyReLU(name=name + '_lrelu4')

        self.output_layer = Dense(n_outputs, kernel_initializer='he_normal', name=name + '_output_layer', trainable=trainable)

    def call(self, inputs):

        x = self.input_layer(inputs)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.lrelu2(x)

        x = self.conv3(x)
        x = self.lrelu3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.lrelu4(x)

        return self.output_layer(x)

    def initialize_weights(self):
        tensor = tf.zeros(self._input_shape)
        _ = self.call(tf.expand_dims(tensor, axis=0))
