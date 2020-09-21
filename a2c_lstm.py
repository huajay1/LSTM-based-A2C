# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class A2CLSTM(object):
    def __init__(
            self,
            sess,
            n_actions,
            n_features,
            lr_a=0.001,
            lr_c=0.01,
            entropy_beta=0.01
    ):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.entroy_beta = entropy_beta

        self.lstm_cell_size = 64

        OPT_A = tf.train.AdamOptimizer(self.lr_a)
        OPT_C = tf.train.AdamOptimizer(self.lr_c)

        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], "state")
            self.a = tf.placeholder(tf.int32, [None, 1], "action")
            self.td_target = tf.placeholder(tf.float32, [None, 1], "td_target")

        self.acts_prob, self.v, self.a_params, self.c_params = self._build_net()

        with tf.name_scope('TD_error'):
            self.td_error = tf.subtract(self.td_target, self.v, name='TD_error')

        with tf.name_scope('c_loss'):
            self.c_loss = tf.reduce_mean(tf.square(self.td_error))

        with tf.name_scope('a_loss'):
            log_prob = tf.reduce_sum(tf.log(self.acts_prob + 1e-5) * tf.one_hot(self.a, self.n_actions, dtype=tf.float32),
                                     axis=1, keepdims=True)
            exp_v = log_prob * tf.stop_gradient(self.td_error)
            entropy = -tf.reduce_sum(self.acts_prob * tf.log(self.acts_prob + 1e-5), axis=1,
                                     keepdims=True)  # encourage exploration
            self.exp_v = self.entroy_beta * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)

        with tf.name_scope('compute_grads'):
            self.a_grads = tf.gradients(self.a_loss, self.a_params)
            self.c_grads = tf.gradients(self.c_loss, self.c_params)

        with tf.name_scope('c_train'):
            self.c_train_op = OPT_C.apply_gradients(zip(self.c_grads, self.c_params))

        with tf.name_scope('a_train'):
            self.a_train_op = OPT_A.apply_gradients(zip(self.a_grads, self.a_params))

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        b_init = tf.constant_initializer(0.1)

        with tf.variable_scope('Critic'):
            # [time_step, feature] => [time_step, batch, feature]
            s = tf.expand_dims(self.s, axis=1, name='timely_input')

            lstm_cell =  tf.nn.rnn_cell.LSTMCell(self.lstm_cell_size)
            self.lstm_state_init = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)

            outputs, _ = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=s,
                initial_state=self.lstm_state_init,
                time_major=True
            )
            cell_out = tf.reshape(outputs[-1, :, :], [-1, self.lstm_cell_size],
                                  name='flatten_lstm_outputs')  # joined state representation

            l_c1 = tf.layers.dense(
                inputs=cell_out,
                units=32,
                activation=tf.nn.tanh,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='l_c1'
            )

            v = tf.layers.dense(
                inputs=l_c1,
                units=1,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='V'
            )  # state value

        with tf.variable_scope('Actor'):
            l_a1 = tf.layers.dense(
                inputs=cell_out,
                units=32,  # number of hidden units
                activation=tf.nn.tanh,
                kernel_initializer=w_init,  # weights
                bias_initializer=b_init,  # biases
                name='l_a1'
            )

            acts_prob = tf.layers.dense(
                inputs=l_a1,
                units=self.n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=w_init,  # weights
                name='acts_prob'
            )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        return acts_prob, v, a_params, c_params

    def choose_action(self, s):
        probs = self.sess.run(self.acts_prob, feed_dict={self.s: s})  # get probabilities for all actions
        a = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return a

    def learn(self, feed_dict):
        self.sess.run([self.a_train_op, self.c_train_op], feed_dict=feed_dict)

    def target_v(self, s):
        v = self.sess.run(self.v, {self.s: s})
        return v
