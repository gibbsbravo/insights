import numpy as np
import random

seed = 34

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

