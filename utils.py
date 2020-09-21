# -*- coding: utf-8 -*-

import itertools
import numpy as np


def action_space(total, ser_num):
    tmp = list(itertools.product(range(total + 1), repeat=ser_num))
    result = []
    for value in tmp:
        if sum(value) == total:
            result.append(list(value))
    result = np.array(result)
    [i, j] = np.where(result == 0)
    result = np.delete(result, i, axis=0)
    print(result.shape)
    return result


def gen_state_(pkt_nums, pos):
    mean = np.array([218.8, 5338, 293])
    std = np.array([51, 847, 42.5])
    state = np.hstack(((pkt_nums - mean) / std, pos))
    return state

# 1:2:3
def gen_state(pkt_nums):
    mean = np.array([218.8, 5338, 293])
    std = np.array([51, 847, 42.5])
    state = (pkt_nums - mean) / std
    return state


def calc__reward(qoe, se):
    qoe_weight = [1, 1, 1]
    se_weight = 0.01
    uility = sum([w * q for w, q in zip(qoe_weight, qoe.tolist())]) + se_weight * se
    if qoe[1] >= 0.98 and qoe[0] >= 0.98:
        if qoe[2] >= 0.95:
            if se < 280:
                reward = 4
            else:
                reward = 4 + (se - 280) * 0.1
        else:
            reward = (qoe[2] - 0.7) * 10
    else:
        reward = -5

    return uility, reward

