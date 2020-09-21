# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import math

class DQN:
    def __init__(
            self,
            sess,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            epsilon_max=0.9,
            epsilon_decay=3000,
            replace_target_iter=300,
            memory_size=2000,
            batch_size=32,
            # e_greedy_increment=None,
            summary_writer=None,
            summary_every=10
    ):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 0 if epsilon_decay is not None else self.epsilon_max
        # self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        self.sess.run(tf.global_variables_initializer())

        if summary_writer is not None:
            # graph was not available when journalist was created
            self.summary_writer = tf.summary.FileWriter(summary_writer)
            self.summary_writer.add_graph(self.sess.graph)
            self.summary_every = summary_every

        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        with tf.name_scope('model_inputs'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
            self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
            self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        # --------------------- build net ---------------------
        with tf.variable_scope('eval_net'):
            self.q_eval = self.q_network(self.s)

        with tf.variable_scope('target_net'):
            self.q_next = self.q_network(self.s_)

        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        # ----------------------- train -----------------------
        with tf.name_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        with tf.name_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope('target_update'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]

        self.summarize = tf.summary.merge_all()

    def q_network(self, s):
        init_w, init_b = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        layer1 = tf.layers.dense(
            inputs=s,
            units=20,
            activation=tf.nn.relu,
            kernel_initializer=init_w,
            bias_initializer=init_b,
            name='layer1'
        )

        q_val = tf.layers.dense(
            inputs=layer1,
            units=self.n_actions,
            kernel_initializer=init_w,
            bias_initializer=init_b,
            name='q_val'
        )
        return q_val

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.epsilon = max(0, self.epsilon_max - math.exp(-1 * self.learn_step_counter / self.epsilon_decay))
        self.learn_step_counter += 1
