
import numpy as np
import os
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math
import scipy.io as io
import scipy
import torch
import torch.nn.functional as F
import h5py
from libtiff import TIFF
from scipy.interpolate import griddata

# import ModelProjectionFourier3_v1 as Model_PJ
# import ModelAberration_Gpu_v4 as Modle_A_G


def Initialize_pathsmesh_interp_unwarp(PhantomSize):
    """
    初始化插值路径 Initialize the interpolation path
    :param PhantomSize: 强度矩阵尺寸 Intensity matrix size
    :return: 初始化插值路径 Initialize the interpolation path
    """
    X_order = np.linspace(-1, 1, PhantomSize[0]).astype(np.float32)
    Y_order = np.linspace(-1, 1, PhantomSize[1]).astype(np.float32)
    Z_order = np.linspace(-1, 1, PhantomSize[2]).astype(np.float32)

    [Phantom_X, Phantom_Y, Phantom_Z] = np.meshgrid(X_order, Y_order, Z_order)

    mesh = np.stack([Phantom_X, Phantom_Y, Phantom_Z], 3)
    mesh = mesh[np.newaxis, :]

    mesh = mesh[:, :, :, :, [2, 0, 1]]

    return mesh


def Func_BackwardEstimation(pathsmesh_interp, param):
    """
    生成逆向插值矩阵 Generate inverse interpolation matrix
    :param pathsmesh_interp: 正向插值矩阵 Forward interpolation matrix
    :param param: 参数 parameter
    :return: 逆向插值矩阵 Inverse interpolation matrix
    """


    # time_1 = time()
    PhantomSize = param.PhantomSize
    pathsmesh_interp_unwarp = torch.from_numpy(Initialize_pathsmesh_interp_unwarp(PhantomSize)).to(param.device)

    ShiftMap = pathsmesh_interp - pathsmesh_interp_unwarp
    del pathsmesh_interp
    torch.cuda.empty_cache()

    # for i in range(0, 3):
    #     ShiftMap_numpy = np.squeeze(ShiftMap[:, :, :, :, i].cpu().detach().numpy())
    #     Addr = SavePath + 'ShiftMap_numpy' + str(i) + str(i_RotAngle)
    #     tif = TIFF.open(Addr + '.tif', mode='w')
    #     for i in range(0, ShiftMap_numpy.shape[2]):
    #         tif.write_image(np.squeeze(ShiftMap_numpy[:, :, i]))
    #     tif.close()

    pathsmesh_interp_r = pathsmesh_interp_unwarp - ShiftMap

    # torch.cuda.empty_cache()

    for i_dem in range(0, 3):
        Temp_ShiftMap = ShiftMap[:, :, :, :, i_dem].reshape(1, 1, PhantomSize[0], PhantomSize[1], PhantomSize[2])
        # Temp_ShiftMap = Temp_ShiftMap.permute(0, 1, 2, 3, 4)
        Temp_ShiftMap = F.grid_sample(Temp_ShiftMap, pathsmesh_interp_r,
                                      mode='bilinear',
                                      padding_mode='zeros',
                                      align_corners=True)
        # Temp_ShiftMap = Temp_ShiftMap.permute(0, 1, 2, 3, 4)
        ShiftMap[0, :, :, :, i_dem] = Temp_ShiftMap.reshape(1, PhantomSize[0], PhantomSize[1], PhantomSize[2])

    del Temp_ShiftMap, pathsmesh_interp_r
    torch.cuda.empty_cache()

    # for i in range(0, 3):
    #     ShiftMap_numpy = np.squeeze(ShiftMap[:, :, :, :, i].cpu().detach().numpy())
    #     Addr = SavePath + 'ShiftMap_2_numpy' + str(i) + str(i_RotAngle)
    #     tif = TIFF.open(Addr + '.tif', mode='w')
    #     for i in range(0, ShiftMap_numpy.shape[2]):
    #         tif.write_image(np.squeeze(ShiftMap_numpy[:, :, i]))
    #     tif.close()

    pathsmesh_interp_unwarp = pathsmesh_interp_unwarp - ShiftMap

    # UnWrappedPhantom_X_gpu = F.grid_sample(WrappedPhantom_X_gpu, pathsmesh_interp_unwarp,
    #                                        mode='bilinear',
    #                                        padding_mode='zeros',
    #                                        align_corners=True)
    # print(time() - time_1)
    return pathsmesh_interp_unwarp
