#!/usr/bin/env python3
"""inverse time decay"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """leaarning rate"""
    learn = alpha / (1 + decay_rate * (global_step // decay_step))
    return learn
