# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class A2C(object):
    def __init__(
            self,
            sess,
            n_actions,
            n_features,
            lr_a=0.001,
            lr_c=0.01,
            reward_decay=0.9
    ):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = reward_decay

        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], "state")
            self.a = tf.placeholder(tf.int32, None, "action")
            self.r = tf.placeholder(tf.float32, None, 'reward')
            self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")

        self.acts_prob, self.v = self._build_net()

        with tf.name_scope('TD_error'):
            self.td_error = self.r + self.gamma * self.v_ - self.v

        with tf.name_scope('c_loss'):
            self.c_loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval

        with tf.name_scope('a_loss'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.a_loss = -tf.reduce_mean(log_prob * tf.stop_gradient(self.td_error))  # advantage (TD_error) guided loss

        with tf.name_scope('c_train'):
            self.c_train_op = tf.train.AdamOptimizer(self.lr_c).minimize(self.c_loss)

        with tf.name_scope('a_train'):
            self.a_train_op = tf.train.AdamOptimizer(self.lr_a).minimize(self.a_loss)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        b_init = tf.constant_initializer(0.1)

        with tf.variable_scope('Actor'):
            l_a = tf.layers.dense(
                inputs=self.s,
                units=32,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=w_init,  # weights
                bias_initializer=b_init,  # biases
                name='l_a'
            )

            acts_prob = tf.layers.dense(
                inputs=l_a,
                units=self.n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=w_init,  # weights
                bias_initializer=b_init,  # biases
                name='acts_prob'
            )

        with tf.variable_scope('Critic'):
            l_c = tf.layers.dense(
                inputs=self.s,
                units=32,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=w_init,  # weights
                bias_initializer=b_init,  # biases
                name='l_c'
            )

            v = tf.layers.dense(
                inputs=l_c,
                units=1,  # output units
                activation=None,
                kernel_initializer=w_init,  # weights
                bias_initializer=b_init,  # biases
                name='V'
            )

        return acts_prob, v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int

    def learn(self, s, a, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        feed_dict = {self.s: s, self.a: a, self.v_: v_, self.r: r}
        self.sess.run([self.a_train_op, self.c_train_op], feed_dict=feed_dict)
