import numpy as np
import os
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math
import scipy.io as io
import torch
import torch.nn.functional as F
import h5py
from libtiff import TIFF



class parameter:
    def __init__(self):
        # device
        self.device = []

def Func_InitializeAberrationParameter(param_meta):
    """
    初始化像差参数 Initialize aberration parameters
    :param param_meta: 初始化参数 Initialization parameters
    :return: 像差参数 Aberration parameters
    """
    rate = param_meta.rate
    ## Initialize_Ray

    ## rotation
    center = [0.5, 0.5, 0.5]

    ## Initialize_RK4
    RK4Range = 7
    Num_RaySample = int(200 / rate)

    ## param_aberration
    param_aberration = parameter()

    param_aberration.dthetaZ = param_meta.dthetaZ
    param_aberration.dthetaY = param_meta.dthetaY
    param_aberration.dthetaX = param_meta.dthetaX

    param_aberration.FlagGrad_RefractionIndex = False
    param_aberration.device = param_meta.device_Aberration

    param_aberration.Num_Ray = param_meta.Num_Ray
    param_aberration.center = center
    # param_aberration.ZnEqRate = param_meta.ZnEqRate

    param_aberration.Num_RaySample = Num_RaySample
    param_aberration.RK4Range = RK4Range

    param_aberration.A_base = param_meta.A_base[0]
    param_aberration.Num_RIGuassKernel = param_meta.Num_RIGuassKernel

    param_aberration.PhantomSize = param_meta.PhantomSize

    return param_aberration