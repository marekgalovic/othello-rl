import tensorflow as tf
from tensorflow.keras import layers


def residual_conv2d(filters, kernel_size, activation=tf.nn.relu, name=None):
    conv1 = layers.Conv2D(
        filters,
        kernel_size,
        padding='same'
    )
    bn1 = layers.BatchNormalization()

    conv2 = layers.Conv2D(
        filters,
        kernel_size,
        padding='same'
    )
    bn2 = layers.BatchNormalization()

    return layers.Lambda(lambda x: activation(tf.add(x, bn2(conv2(activation(bn1(conv1(x))))))), name=name)


class Agent(tf.keras.Model):

    def __init__(self, board_size, hidden_size=32, num_residual_conv=3):
        super(Agent, self).__init__()

        self._convolutions = [
            layers.Conv2D(filters=hidden_size, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='input_conv')
        ] + [
            residual_conv2d(filters=hidden_size, kernel_size=(3,3), name='residual_conv_%d' % i)
            for i in range(num_residual_conv)
        ]

        self._flatten = layers.Flatten()
        self._value_conv = layers.Conv2D(filters=2*hidden_size, kernel_size=(3,3), activation=tf.nn.relu, name='value_conv')
        self._policy_conv = layers.Conv2D(filters=2*hidden_size, kernel_size=(3,3), activation=tf.nn.relu, name='policy_conv')

        self._value = layers.Dense(1, name='value')
        self._policy = layers.Dense(board_size ** 2, name='policy')

    def call(self, state):
        conv = state
        for conv_layer in self._convolutions:
            conv = conv_layer(conv)

        value_conv = self._flatten(self._value_conv(conv))
        policy_conv = self._flatten(self._policy_conv(conv))

        value = tf.squeeze(self._value(value_conv), -1)
        policy = self._policy(policy_conv)

        return tf.nn.softmax(policy), value
