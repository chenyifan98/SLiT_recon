import sys
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
import copy

import Func_IntensityOpt_v40_nAnlge as Func_IntensityRecon


from Func_TiffStackSave import TiffStackSave as TiffStackSave
from Func_TiffStackSave import TiffStackLoad as TiffStackLoad


import Func_zernike_v1 as Func_zernike





class parameter:
    """
    用于传递参数
    Used to pass parameters
    """
    def __init__(self):
        # device
        self.device = []


def Initialize_Refraction(Num_RIGuassKernel, A_base, Sig_base):
    """
    其他项目的共用代码，这里并没有实际意义
    Shared code from other projects, no real significance here
    :param Num_RIGuassKernel:
    :param A_base_set:
    :param Sig_base:
    :return:
    """
    sig = Sig_base * np.ones([Num_RIGuassKernel, Num_RIGuassKernel, Num_RIGuassKernel])
    sig_swap = sig

    A = A_base[0] * np.ones([Num_RIGuassKernel, Num_RIGuassKernel, Num_RIGuassKernel])

    # io.savemat('asig', {'sig': sig, 'A': A})

    A_swap = np.swapaxes(A, 0, 1)

    A = A_swap  # [y, x, z]
    sig = sig_swap  # [y, x, z]

    RefractionIndex = np.stack([A, sig], axis=3)

    return RefractionIndex


def Initialize_ZernikeItem(ZernikeRank, Num_Ray):
    """
    初始化泽尼克项
    Initialize Zernike terms
    :param ZernikeRank: Zernike级数 Zernike series
    :param Num_Ray: 采样数量 Number of samples
    :return: 初始化的射线参数矩阵 Initialized ray parameter matrix
    """
    # ZernikeRank = 1
    ZernikeItem = Func_zernike.zernike_dictionary(ZernikeRank, [Num_Ray, Num_Ray])

    ZernikeItem = ZernikeItem.astype('float32')
    ZernikeItem = torch.from_numpy(ZernikeItem)

    return ZernikeItem


def Initialize_Ray(Num_Ray):
    """
    初始化射线采样点的位置信息
    Initialize the position information of the ray sampling point
    :param Num_Ray: 采样数量 Number of samples
    :return: 初始化的射线参数矩阵 Initialized ray parameter matrix
    """

    # Num_Ray = 101
    X_coord = np.linspace(0, 1, Num_Ray)
    Y_coord = np.linspace(0, 1, Num_Ray)
    X_coord, Y_coord = np.meshgrid(X_coord, Y_coord)

    xyz_init0 = np.zeros([Num_Ray, Num_Ray, 5])
    xyz_init0[:, :, 0] = X_coord
    xyz_init0[:, :, 1] = 0
    xyz_init0[:, :, 2] = Y_coord
    xyz_init0[:, :, 3] = 0
    xyz_init0[:, :, 4] = 0

    xyz_init0 = xyz_init0.astype('float32')

    return xyz_init0


def Func_InitAvgMusk(PhantomSize):
    """
    初始化强度矩阵
    Initialize the intensity matrix
    :param PhantomSize: 矩阵大小 Matrix size
    :return: 初始化强度矩阵 Initialize the intensity matrix
    """

    musk = np.zeros(PhantomSize)
    Xc = np.linspace(0, 0, PhantomSize[0], dtype='float32')
    Yc = np.linspace(0, 0, PhantomSize[1], dtype='float32')
    Zc = np.linspace(1, 0, PhantomSize[2], dtype='float32')
    Yc, Xc, musk = np.meshgrid(Xc, Yc, Zc)
    musk = musk*2

    return musk


def Func_SaveYprojection(Phantom_List, SavePath, iter):
    """
    保存整个list的Y向投影
    Save the Y projection of the entire list
    :param Phantom_List: 强度矩阵list intensity matrix list
    :param SavePath: 保存路径 Save Path
    :param iter: 迭代次数 Iterations number
    :return:
    """

    Phantom_List_Y = torch.max(Phantom_List, 2)[0].squeeze().permute([1, 2, 0])

    addr = SavePath + 'PhantomY_' + str(iter)
    TiffStackSave(Phantom_List_Y, addr)

    return


if __name__ == '__main__':
    # PSF 路径
    PSF_addr = '../../PSF/psf_sim_5x0.14_121_121_15_15_721d6_v73_210731/psf_sim_5x0.14_121_121_15_15_721d6_f0_v73.mat'
    # 成像结果路径
    WignerPath = '../../Wigner_Sim/210629/fishR_ballG_1750_6_ReOrder_ReAlign15d6_Cut721_Cx956Cy956_1'
    # 波前导入路径
    WaveFrontPath = './Recon/PipeLineOptSync_v31x_f0_noSync_BG150/WaveFront/RayInitList_20'
    # WaveFrontPath = 0
    PhantomPath = 0
    # SaveAddr
    SavePath_root_up = './Recon/'
    if not os.path.exists(SavePath_root_up): os.mkdir(SavePath_root_up)
    SavePath_root = SavePath_root_up + 'PipeLineOptSync_v34x_f0_24Angle_1/'
    SavePath_test_root = SavePath_root[0:-1] + '_test/'
    if not os.path.exists(SavePath_root): os.mkdir(SavePath_root)
    if not os.path.exists(SavePath_test_root): os.mkdir(SavePath_test_root)

    #### META PARAMETERS
    WignerSync = False                                                  # 是否同步更新 wigner Whether to update wigner synchronously
    MagnifyRate = 1                                                     # 强度放大倍率 Intensity magnification
    StepRate = 0.8                                                      # RL优化步长系数 Optimize step size coefficient
    BackGround = 100                                                    # 成像背景强度 Imaging background intensity

    GHWFMX = 0.0001                                                     # X维度模糊化系数 X-dimension fuzzification coefficient
    GHWFMY = 0.0001                                                     # Y维度模糊化系数 Y-dimension fuzzification coefficient
    GHWFMZ = 0.0001                                                     # Z维度模糊化系数 Z-dimension fuzzification coefficient

    RealSizeSingle = 721                                                # 强度重建体模大小 Intensity reconstruction size

    param_meta = parameter()
    param_meta.IterNum = 20                                             # 整体旋转数量 The total number of iterations
    param_meta.FlagTestSave = True                                      # 测试点保存标记 Test point save mark
    param_meta.RotAngle = 24                                            # 旋转角度 Number of rotations
    param_meta.RotAngleSkip = 1 

    param_meta.dthetaZ = 0                                              # Z方向旋转角度 1 Z-axis rotation angle
    param_meta.dthetaY = 0                                              # Y方向旋转角度 2 Y-axis rotation angle
    param_meta.dthetaX = -math.pi * 2 / param_meta.RotAngle             # X方向旋转角度 3 X-axis rotation angle


    param_meta.PhantomBackground = 1e2
    param_meta.PhantomMaxPersent = 0.01
    param_meta.RealSize = [RealSizeSingle, RealSizeSingle, RealSizeSingle]# 强度重建体模大小 Intensity reconstruction size
    param_meta.AddEdge = [0, 0, 0]                                      # 边界扩展 Boundary Extension
    param_meta.PhantomSize = list(np.array(param_meta.RealSize)+np.array(param_meta.AddEdge) * 2)
                                                                        # 强度重建体模大小 Intensity reconstruction phantom size
    param_meta.ZernikeRank = 2                                          # 波前zernike项阶数 Wavefront zernike term order
    param_meta.ZernikeItemNum = int((param_meta.ZernikeRank+2)*(param_meta.ZernikeRank+1)/2)
                                                                        # 波前zernike项项数 Wavefront zernike item number
    param_meta.rate = 8                                                 # 折射率体模降采样倍数（无关系数）Refractive index phantom downsampling factor(Irrelevant coefficient)
    param_meta.Num_RIGuassKernel = int(128 / param_meta.rate)           # 输入折射率矩阵大小（无关系数） Input refractive index matrix size(Irrelevant coefficient)
    param_meta.A_base = [1.3, 1.31]                                     # 折射率矩阵基础强度（无关系数）Refractive Index Matrix Base Strength(Irrelevant coefficient)
    param_meta.Sig_base = 0.005 * param_meta.rate                       # 折射率矩阵基础扩散范围 （无关系数）Refractive Index Matrix Base Diffuse Range(Irrelevant coefficient)
    param_meta.Num_Ray = int(160/param_meta.rate + 1)                   # 光追线数量 Number of rays


    ## Initialize_Phantom
    Phantom_numpy = np.zeros(param_meta.PhantomSize)+1
    if PhantomPath != 0:
        Phantom_numpy = TiffStackLoad(param_meta.RealSize, PhantomPath)

    Phantom = Phantom_numpy.astype('float32')
    Phantom_init = torch.from_numpy(Phantom)
    Phantom = torch.from_numpy(Phantom)
    Phantom_ = Phantom * 1
    Phantom_AVG = Phantom * 1
    PhantomSize = list(Phantom_numpy.shape)


    ## Loading Projrction
    data = io.loadmat('../../WJM_Map/wjm_chyf15_map.mat')
    map_Intensity = data['map']

    ## Initialize_PSF
    mat = h5py.File(PSF_addr, 'r')
    PSF_numpy = np.transpose(mat['psf'])
    PSF_numpy = PSF_numpy.astype('float32')
    PSF = torch.from_numpy(PSF_numpy)
    PSFSize = list(PSF_numpy.shape)

    ## Initialize_Refraction
    RefractionIndex = Initialize_Refraction(param_meta.Num_RIGuassKernel, param_meta.A_base, param_meta.Sig_base)
    RefractionIndex = RefractionIndex.astype('float32')
    RefractionIndex = torch.from_numpy(RefractionIndex)


    ## Initialize_WaveFront
    # RayInit Base Initialize
    RayInitList0 = np.zeros([param_meta.RotAngle, param_meta.Num_Ray, param_meta.Num_Ray, 5])
    for i in range(0, param_meta.RotAngle):
        RayInitList0[i, :, :, :] = Initialize_Ray(param_meta.Num_Ray)
    RayInitList0 = RayInitList0.astype('float32')
    RayInitList0 = torch.from_numpy(RayInitList0)
    # ZernikeItem
    ZernikeItem = Initialize_ZernikeItem(param_meta.ZernikeRank, param_meta.Num_Ray)
    # Zernike Shift Initialize
    ZernikeItemNum = param_meta.ZernikeItemNum
    RayInitZernikeList = np.zeros([param_meta.RotAngle, ZernikeItemNum, 5])
    RayInitZernikeList = RayInitZernikeList.astype('float32')
    RayInitZernikeList = torch.from_numpy(RayInitZernikeList)
    # RayInitList Initialize
    RayInitList = np.zeros([param_meta.RotAngle, param_meta.Num_Ray, param_meta.Num_Ray, 5])
    RayInitList = RayInitList.astype('float32')
    RayInitList = torch.from_numpy(RayInitList)
    for i in range(0, param_meta.RotAngle):
        for j in range(0, 5):
            RayInitList[i, :, :, j] = torch.sum(ZernikeItem * RayInitZernikeList[i, :, j], 2)
    RayInitList = RayInitList + RayInitList0

    if WaveFrontPath != 0:
        data = io.loadmat(WaveFrontPath)
        RayInitList = data['RayInitList']
        RayInitList = torch.from_numpy(RayInitList)


    #### PARAMETERS ####

    #### RLRecon PARAMETERS
    param_RLRecon = parameter()
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    param_RLRecon.device = device
    param_RLRecon.device_Aberration = device
    device = torch.device("cpu")
    param_RLRecon.deviceOut = device

    param_RLRecon.EpochNum = 1
    param_RLRecon.StepRate = StepRate
    param_RLRecon.map = map_Intensity

    param_RLRecon.RotAngle = param_meta.RotAngle
    param_RLRecon.RotAngleSkip = param_meta.RotAngleSkip


    param_RLRecon.MagnifyRate = MagnifyRate
    param_RLRecon.WignerSync = WignerSync
    param_RLRecon.center = [0.5, 0.5, 0.5]
    param_RLRecon.PhantomSize = PhantomSize
    param_RLRecon.threshold = 1e-4
    param_RLRecon.CompressRate = 11
    param_RLRecon.AvgMusk = Func_InitAvgMusk(PhantomSize)
    param_RLRecon.CutEdge = [10, 10, 10] # y,x,z
    param_RLRecon.BackGround = BackGround
    param_RLRecon.AddEdge = param_meta.AddEdge

    param_RLRecon.PSFSize = PSFSize

    param_RLRecon.dthetaZ = param_meta.dthetaZ
    param_RLRecon.dthetaY = param_meta.dthetaY
    param_RLRecon.dthetaX = param_meta.dthetaX

    param_RLRecon.GHWFMX = GHWFMX
    param_RLRecon.GHWFMY = GHWFMY
    param_RLRecon.GHWFMZ = GHWFMZ

    param_RLRecon.FlagAberration = True
    param_RLRecon.rate = param_meta.rate
    param_RLRecon.A_base = param_meta.A_base
    param_RLRecon.Sig_base = param_meta.Sig_base
    param_RLRecon.Num_Ray = param_meta.Num_Ray
    param_RLRecon.Num_RIGuassKernel = param_meta.Num_RIGuassKernel

    param_RLRecon.Flag_AngleSave = False
    param_RLRecon.FlagTestSave = param_meta.FlagTestSave
    param_RLRecon.WignerPath = WignerPath


    #### ShiftShiftMap PARAMETERS
    param_ShiftShiftMap = parameter()
    device = torch.device("cuda:0")
    param_ShiftShiftMap.device = device
    param_ShiftShiftMap.PhantomSize = PhantomSize
    param_ShiftShiftMap.Flag_check = 0


    ## IterTrain
    for iter in range(1, param_meta.IterNum+1):
        param_RLRecon.BigEpoch = iter

        SavePath = SavePath_root + 'Intensity/'
        if not os.path.exists(SavePath): os.mkdir(SavePath)
        param_RLRecon.SavePath = SavePath

        SavePath = SavePath_test_root + 'Intensity/'
        if not os.path.exists(SavePath): os.mkdir(SavePath)
        param_RLRecon.SavePathTest = SavePath

        [Phantom_List, Phantom_AVG, warped_Phantom_List] = Func_IntensityRecon.RotationAberrationSyncErrorRLReconstructionFourier3(
            param_RLRecon, param_ShiftShiftMap, Phantom_AVG, RefractionIndex, RayInitList, PSF)


        SavePath = SavePath_root + 'Intensity_Yprojection/'
        if not os.path.exists(SavePath): os.mkdir(SavePath)
        Func_SaveYprojection(Phantom_List, SavePath, iter)





