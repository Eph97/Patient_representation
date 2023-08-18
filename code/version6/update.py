import matplotlib.pyplot as plt
import numpy as np

def C_2(x, bar_x):
    return ((1 + x*bar_x) / (1 + bar_x**2)).reshape(-1,1)
    # return np.dot(x, bar_x) / np.dot(bar_x, bar_x)
    # return x / bar_x

def C_cov(x, bar_x, v1, v2, cov):
    return ((v1 + (x + bar_x)*cov + x*bar_x*v2) / (v1 +2*bar_x*cov + v2* bar_x**2)).reshape(-1,1)
    # return np.dot(x, bar_x) / np.dot(bar_x, bar_x)
    # return x / bar_x

def C(x, bar_x):
    # return ((1 + x*bar_x) / (1 + bar_x**2)).reshape(-1,1)
    return np.dot(x, bar_x) / np.dot(bar_x, bar_x)
    # return x / bar_x


def C_multi(x1,x2, xb1,xb2, v0, v1, v2):
    return (v0 + x1*xb1*v1 + x2*xb2*v2)/(v0 + xb1**2*v1 + xb2**2*v2)
