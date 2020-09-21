# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import copy
import os
from a2c_lstm import A2CLSTM
from env import EnvMove
import utils

# QoE_WEIGHT = [1, 1, 1]
# SE_WEIGHT = 0.01
UE_NUMS = 1200
SER_PROB = [1, 2, 3]
LEARNING_WINDOW = 2000
BAND_WHOLE = 10  # M
BAND_PER = 0.2  # M
DL_MIMO = 64
SER_CAT = ['volte', 'embb_general', 'urllc']

LR_A = 0.002
LR_C = 0.01
GAMMA = 0
ENTROY_BETA = 0.001
LSTM_LEN = 10
MAX_ITERATIONS = 10000

LOG_TRAIN = './logs/a2clstm.txt'
# LOG_TRAIN = './logs/a2c.txt'

action_space = utils.action_space(int(BAND_WHOLE // BAND_PER), len(SER_CAT)) * BAND_PER * 10 ** 6
n_actions = len(action_space)
print(n_actions)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = A2CLSTM(sess, n_features=len(SER_CAT), n_actions=n_actions, lr_a=LR_A, lr_c=LR_C, entropy_beta=ENTROY_BETA)

env = EnvMove(UE_max_no=UE_NUMS, ser_prob=np.array(SER_PROB, dtype=np.float32), learning_windows=LEARNING_WINDOW, dl_mimo=DL_MIMO)

qoe_lst, se_lst = [], []
reward_lst = []

buffer_ob = []

for i in range(LSTM_LEN):
    env.countReset()
    env.user_move()
    env.activity()

    action = np.random.choice(n_actions)
    env.band_ser_cat = action_space[action]

    for i_subframe in range(LEARNING_WINDOW):
        env.scheduling()
        env.provisioning()
        if i_subframe < LEARNING_WINDOW - 1:
            env.activity()

    pkt, dis = env.get_state()
    observe = utils.gen_state(pkt)
    buffer_ob.append(observe)

for i_iter in range(MAX_ITERATIONS):
    env.countReset()
    env.user_move()
    env.activity()

    s = np.vstack(buffer_ob)
    action = model.choose_action(s)
    env.band_ser_cat = action_space[action]

    for i_subframe in range(LEARNING_WINDOW):
        env.scheduling()
        env.provisioning()
        if i_subframe < LEARNING_WINDOW - 1:
            env.activity()

    pkt, dis = env.get_state()
    observe = utils.gen_state(pkt)
    buffer_ob.pop(0)
    buffer_ob.append(observe)
    s_ = np.vstack(buffer_ob)

    qoe, se = env.get_reward()
    qoe_lst.append(qoe.tolist())
    se_lst.append(se[0])

    uility, reward = utils.calc__reward(qoe, se[0])
    reward_lst.append(reward)

    print('\nStep-%d' % i_iter)
    print('qoe: ', qoe)
    print('se: ', se[0])
    print('reward: ', reward)

    v_s_ = model.target_v(s_)
    td_target = reward + GAMMA * v_s_

    feed_dict = {
        model.s: s,
        model.a: np.vstack([action]),
        model.td_target: np.vstack([td_target])
    }

    model.learn(feed_dict)

    if (i_iter + 1) % 500 == 0:
        with open(LOG_TRAIN, 'a+') as f:
            for i in range(len(se_lst)):
                print(
                    'Reward: %.4f, SE: %.4f, QoE_volte: %.4f, QoE_embb: %.4f, QoE_urllc: %.4f' % (
                        reward_lst[i], se_lst[i], qoe_lst[i][0], qoe_lst[i][1], qoe_lst[i][2]), file=f)
        qoe_lst, se_lst = [], []
        reward_lst = []
