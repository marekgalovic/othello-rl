import tensorflow as tf
from tensorflow.keras import layers


def gather_action_probabilities(p, action_ids):
    gather_indices = tf.stack([
        tf.range(tf.shape(action_ids)[0]),
        action_ids
    ], -1)

    return tf.gather_nd(p, gather_indices)


def ppo_loss(new_values, values, p, p_old, action_ids, rewards, eps=0.2, c=1.0):
    advantage = rewards - values

    p = gather_action_probabilities(p, action_ids)
    r = p / p_old

    l_pg = tf.reduce_min([r * advantage, tf.clip_by_value(r, 1-eps, 1+eps) * advantage], axis=0)
    l_v = tf.square(new_values - rewards)

    return tf.reduce_mean(-l_pg + c*l_v)


def ce_loss(action_p, state_v, action_ids, rewards):
    policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        tf.one_hot(action_ids, tf.shape(action_p)[1]),
        action_p,
    ))
    value_loss = tf.reduce_mean(tf.square(rewards - state_v))

    return policy_loss + value_loss


def train(agent, optimizer, states, old_action_p, action_indices, state_values, rewards):
    with tf.GradientTape() as t:
        action_p, new_state_values = agent(states)

        # loss = ce_loss(action_p, new_state_values, action_indices, rewards)
        loss = ppo_loss(new_state_values, state_values, action_p, old_action_p, action_indices, rewards)

    grads = t.gradient(loss, agent.trainable_variables)
    optimizer.apply_gradients(zip(grads, agent.trainable_variables))

    return loss


class Agent(tf.keras.Model):

    def __init__(self, board_size, hidden_size=256, num_conv=5):
        super(Agent, self).__init__()

        self._flatten = layers.Flatten()

        self._convolutions = [
            layers.Conv2D(filters=hidden_size, kernel_size=(3,3), padding='same', name='conv_%d' % i)
            for i in range(num_conv)
        ]
        self._conv_batch_norms = [
            layers.BatchNormalization(name='bn_%d' % i)
            for i in range(num_conv)
        ]

        self._hidden1 = layers.Dense(2 * hidden_size, name='hidden1')
        self._hidden1_bn = layers.BatchNormalization(name='hidden1_bn')

        self._hidden2 = layers.Dense(hidden_size, name='hidden2')
        self._hidden2_bn = layers.BatchNormalization(name='hidden2_bn')

        self._policy = layers.Dense(board_size ** 2, name='policy')
        self._value = layers.Dense(1, name='value')

    def call(self, state, raw_pi=False):
        for i, conv_layer in enumerate(self._convolutions):
            bn = self._conv_batch_norms[i]
            state = tf.nn.relu(bn(conv_layer(state)))

        state = self._flatten(state)

        state = tf.nn.relu(self._hidden1_bn(self._hidden1(state)))
        state = tf.nn.relu(self._hidden2_bn(self._hidden2(state)))

        policy = self._policy(state)
        value = tf.nn.tanh(self._value(state))
        value = tf.squeeze(value, -1)

        if raw_pi:
            return policy, value

        return tf.nn.softmax(policy), value


# def residual_conv2d(filters, kernel_size, activation=tf.nn.relu, name=None):
#     conv1 = layers.Conv2D(
#         filters,
#         kernel_size,
#         padding='same'
#     )
#     bn1 = layers.BatchNormalization()

#     conv2 = layers.Conv2D(
#         filters,
#         kernel_size,
#         padding='same'
#     )
#     bn2 = layers.BatchNormalization()

#     return layers.Lambda(lambda x: activation(tf.add(x, bn2(conv2(activation(bn1(conv1(x))))))), name=name)


# class AgentV2(tf.keras.Model):

#     def __init__(self, board_size, hidden_size=256, num_residual_conv=5, dropout=0.0):
#         super(Agent, self).__init__()

#         self._convolutions = [
#             layers.Conv2D(filters=hidden_size, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='input_conv')
#         ] + [
#             residual_conv2d(filters=hidden_size, kernel_size=(3,3), name='residual_conv_%d' % i)
#             for i in range(num_residual_conv)
#         ]

#         self._flatten = layers.Flatten()
#         self._dropout = layers.Dropout(dropout)

#         self._policy_conv = layers.Conv2D(filters=2, kernel_size=(1,1), name='policy_conv')
#         self._bn_policy = layers.BatchNormalization()

#         self._value_conv = layers.Conv2D(filters=1, kernel_size=(1,1), name='value_conv')
#         self._bn_value = layers.BatchNormalization()

#         self._policy = layers.Dense(board_size ** 2, name='policy')
#         self._value_hidden = layers.Dense(hidden_size, name='value')
#         self._value = layers.Dense(1, name='value')

#     def call(self, state, raw_pi=False):
#         conv = state
#         for conv_layer in self._convolutions:
#             conv = self._dropout(conv_layer(conv))

#         policy_conv = self._dropout(self._flatten(tf.nn.relu(self._bn_policy(self._policy_conv(conv)))))
#         value_conv = self._dropout(self._flatten(tf.nn.relu(self._bn_value(self._value_conv(conv)))))

#         policy = self._policy(policy_conv)
#         value = self._dropout(tf.nn.relu(self._value_hidden(value_conv)))
#         value = tf.nn.tanh(self._value(value))
#         value = tf.squeeze(value, -1)

#         if raw_pi:
#             return policy, value

#         return tf.nn.softmax(policy), value
