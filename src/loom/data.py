import numpy as np


class Data:
    def __init__(self, x_b, y_b_target=None):
        self.x_b = x_b
        self.batch_size = x_b.shape[0]
        self.y_b_target = y_b_target
