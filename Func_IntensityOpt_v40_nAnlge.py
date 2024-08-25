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

import Func_ZCPFPBP_v1 as Func_FPBP
import ModelAberration_Rot_v4 as Modle_A_G
import Func_InitializeAberrationParameter_v2 as InitAberParam
import Func_ShiftShiftMap as Func_DeAberration
from Func_TiffStackSave import TiffStackSave as TiffStackSave


def Theta2TransMatrix(param):
    """
    旋转角度转换到旋转矩阵 Convert rotation angle to rotation matrix
    :param param: 参数 parameter
    :return: 更新后的参数 Updated parameters
    """

    theta = param.thetaY
    center = param.center
    RotationMatrix_f = np.matrix([[math.cos(theta), -math.sin(theta), 0],
                              [math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])
    param.TransMatrix_Y = torch.from_numpy(RotationMatrix_f).float().to(param.device)

    theta = -param.thetaX
    center = param.center
    RotationMatrix_f = np.matrix([[math.cos(theta), -math.sin(theta), 0],
                              [math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])
    param.TransMatrix_X = torch.from_numpy(RotationMatrix_f).float().to(param.device)

    theta = param.thetaZ
    center = param.center
    RotationMatrix_f = np.matrix([[math.cos(theta), -math.sin(theta), 0],
                              [math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])
    param.TransMatrix_Z = torch.from_numpy(RotationMatrix_f).float().to(param.device)
    return param


def Rotation_Y(X, param):
    """
    Y轴旋转矩阵 Y-axis rotation matrix
    :param X: 强度矩阵 intensity Matrix
    :param param: 旋转参数 Rotation parameters
    :return: 旋转后的强度矩阵 Rotated intensity matrix
    """

    coor_1 = torch.linspace(-1, 1, param.PhantomSize[0]).to(param.device)
    coor_2 = torch.linspace(-1, 1, param.PhantomSize[2]).to(param.device)

    [coor_1, coor_2] = torch.meshgrid(coor_1, coor_2)

    temp_ones = torch.ones([param.PhantomSize[0], param.PhantomSize[2]]).to(param.device)

    coor = torch.stack([coor_2, coor_1, temp_ones], axis=2)

    coor = coor.reshape([param.PhantomSize[0]*param.PhantomSize[2], 3])
    coor = coor.mm(param.TransMatrix_Y)
    coor = coor.reshape([param.PhantomSize[0], param.PhantomSize[2], 3])

    coor = coor[:, :, 0:2].reshape(1, param.PhantomSize[0], param.PhantomSize[2], 2)
    X_rotate = X * 0

    for i_Layer in range(0, param.PhantomSize[0]):
        X_layer = X[i_Layer, :, :].squeeze().reshape(1, 1, param.PhantomSize[1], param.PhantomSize[2])
        # plt.imshow(X_layer.cpu().detach().numpy())
        X_rotate[i_Layer, :, :] = F.grid_sample(X_layer, coor,
                                                mode='bilinear',
                                                padding_mode='zeros',
                                                align_corners=True).squeeze()
    # X_rotate = X_rotate.permute([0, 2, 1])
    return X_rotate


def Rotation_X(X, param):
    """
    X轴旋转矩阵 X-axis rotation matrix
    :param X: 强度矩阵 intensity Matrix
    :param param: 旋转参数 Rotation parameters
    :return: 旋转后的强度矩阵 Rotated intensity matrix
    """

    coor_1 = torch.linspace(-1, 1, param.PhantomSize[0]).to(param.device)
    coor_2 = torch.linspace(-1, 1, param.PhantomSize[2]).to(param.device)

    [coor_1, coor_2] = torch.meshgrid(coor_1, coor_2)

    temp_ones = torch.ones([param.PhantomSize[0], param.PhantomSize[2]]).to(param.device)

    coor = torch.stack([coor_2, coor_1, temp_ones], axis=2)

    coor = coor.reshape([param.PhantomSize[0]*param.PhantomSize[2], 3])
    coor = coor.mm(param.TransMatrix_X)
    coor = coor.reshape([param.PhantomSize[0], param.PhantomSize[2], 3])

    coor = coor[:, :, 0:2].reshape(1, param.PhantomSize[0], param.PhantomSize[2], 2)
    X_rotate = X * 0

    for i_Layer in range(0, param.PhantomSize[1]):
        X_layer = X[:, i_Layer, :].squeeze().reshape(1, 1, param.PhantomSize[0], param.PhantomSize[2])
        # plt.imshow(X_layer.cpu().detach().numpy())
        X_rotate[:, i_Layer, :] = F.grid_sample(X_layer, coor,
                                                mode='bilinear',
                                                padding_mode='zeros',
                                                align_corners=True).squeeze()
    # X_rotate = X_rotate.permute([0, 2, 1])
    return X_rotate


def Rotation_Z(X, param):
    """
    Z轴旋转矩阵 Z-axis rotation matrix
    :param X: 强度矩阵 intensity Matrix
    :param param: 旋转参数 Rotation parameters
    :return: 旋转后的强度矩阵 Rotated intensity matrix
    """
    coor_1 = torch.linspace(-1, 1, param.PhantomSize[0]).to(param.device)
    coor_2 = torch.linspace(-1, 1, param.PhantomSize[2]).to(param.device)

    [coor_1, coor_2] = torch.meshgrid(coor_1, coor_2)

    temp_ones = torch.ones([param.PhantomSize[0], param.PhantomSize[2]]).to(param.device)

    coor = torch.stack([coor_2, coor_1, temp_ones], axis=2)

    coor = coor.reshape([param.PhantomSize[0]*param.PhantomSize[2], 3])
    coor = coor.mm(param.TransMatrix_Z)
    coor = coor.reshape([param.PhantomSize[0], param.PhantomSize[2], 3])

    coor = coor[:, :, 0:2].reshape(1, param.PhantomSize[0], param.PhantomSize[2], 2)
    X_rotate = X * 0

    for i_Layer in range(0, param.PhantomSize[0]):
        X_layer = X[:, :, i_Layer].squeeze().reshape(1, 1, param.PhantomSize[1], param.PhantomSize[2])
        # plt.imshow(X_layer.cpu().detach().numpy())
        X_rotate[:, :, i_Layer] = F.grid_sample(X_layer, coor,
                                                mode='bilinear',
                                                padding_mode='zeros',
                                                align_corners=True).squeeze()
    # X_rotate = X_rotate.permute([0, 2, 1])
    return X_rotate


def Func_Filter_Phantom(Phantom, param):
    """
        对强度矩阵进行模糊化 Fuzzify the intensity matrix
        :param Phantom: 强度矩阵 intensity Matrix
        :param param: 模糊化参数 Fuzzy parameters
        :return: 模糊化强度矩阵 Fuzzification intensity matrix
    """
    PhantomSize = param.PhantomSize
    GstdX = param.GHWFMX / 2.355
    GstdY = param.GHWFMY / 2.355
    GstdZ = param.GHWFMZ / 2.355

    kernel_size = PhantomSize[0]
    m = int((kernel_size - 1.) / 2.)
    y = np.ogrid[-m:m + 1]
    kernel_size = PhantomSize[1]
    m = int((kernel_size - 1.) / 2.)
    x = np.ogrid[-m:m + 1]
    kernel_size = PhantomSize[2]
    m = int((kernel_size - 1.) / 2.)
    z = np.ogrid[-m:m + 1]

    [mx, my, mz] = np.meshgrid(x,y,z)

    h = np.exp(-(my * my) / (2. * GstdY * GstdY)-(mx * mx) / (2. * GstdX * GstdX)-(mz * mz) / (2. * GstdZ * GstdZ)) # 这一步最慢
    sumh = h.sum()
    h /= sumh
    Filter = torch.from_numpy(h.astype('float32')).to(param.device)
    del mx, my, mz, h, sumh
    torch.cuda.empty_cache()

    # h = np.exp(-(x * x) / (2. * GstdZ * GstdZ))
    # sumh = h.sum()
    # h /= sumh
    # h = torch.from_numpy(h).to(param.device)
    #
    # Filter = Phantom* 0
    # Filter[m, m, :] = h

    Filter_fft = torch.fft.fftn(Filter)
    Phantom_fft = torch.fft.fftn(Phantom)

    conv_fft = Filter_fft * Phantom_fft
    del Phantom_fft, Filter_fft, Filter, Phantom
    torch.cuda.empty_cache()

    PhantomFilted = torch.fft.ifftn(conv_fft, s=PhantomSize[0:3])
    PhantomFilted = torch.real(PhantomFilted)

    del conv_fft
    torch.cuda.empty_cache()

    shift_temp = int((kernel_size + 1) / 2)

    # Addr = 'PhantomFilted'
    # PhantomFilted_numpy = np.squeeze(PhantomFilted.cpu().detach().numpy())
    # io.savemat(Addr + '.mat', {'a': 1, 'ImageBack': PhantomFilted_numpy})
    # tif = TIFF.open(Addr + '.tif', mode='w')
    # for i in range(0, PhantomFilted_numpy.shape[2]):
    #     tif.write_image(np.squeeze(PhantomFilted_numpy[:, :, i]))
    # tif.close()

    PhantomFilted = torch.roll(PhantomFilted, shifts=(shift_temp, shift_temp, shift_temp), dims=(0, 1, 2))


    PhantomFilted[torch.isnan(PhantomFilted)] = param.threshold
    PhantomFilted[PhantomFilted < param.threshold] = param.threshold

    # param.PhantomSize_zcp = list(PhantomFilted.size())

    return PhantomFilted, param


def Func_ZCP_Phantom(Phantom, param):
    """
        Z向压缩强度矩阵

        :param Phantom: 强度矩阵
        :param param: 参数
        :return: Z向压缩的强度矩阵
    """
    PhantomSize = param.PhantomSize
    kernel_size = PhantomSize[2]
    CompressRate = param.CompressRate
    std = 2 * CompressRate / 2.355
    m = int((kernel_size - 1.) / 2.)
    x = np.ogrid[-m:m + 1]
    h = np.exp(-(x * x) / (2. * std * std))
    sumh = h.sum()
    h /= sumh
    h = torch.from_numpy(h).to(param.device)


    Filter = Phantom* 0
    Filter[m, m, :] = h

    Filter_fft = torch.fft.fftn(Filter)
    Phantom_fft = torch.fft.fftn(Phantom)

    conv_fft = Filter_fft * Phantom_fft
    del Phantom_fft, Filter_fft, Filter, Phantom
    torch.cuda.empty_cache()

    PhantomFilted = torch.fft.ifftn(conv_fft, s=PhantomSize[0:3])
    PhantomFilted = torch.real(PhantomFilted)
    del conv_fft
    torch.cuda.empty_cache()

    shift_temp = int((kernel_size + 1) / 2)

    # Addr = 'PhantomFilted'
    # PhantomFilted_numpy = np.squeeze(PhantomFilted.cpu().detach().numpy())
    # io.savemat(Addr + '.mat', {'a': 1, 'ImageBack': PhantomFilted_numpy})
    # tif = TIFF.open(Addr + '.tif', mode='w')
    # for i in range(0, PhantomFilted_numpy.shape[2]):
    #     tif.write_image(np.squeeze(PhantomFilted_numpy[:, :, i]))
    # tif.close()

    PhantomFilted = torch.roll(PhantomFilted, shifts=(shift_temp, shift_temp, shift_temp), dims=(0, 1, 2))

    m_ = np.floor((kernel_size - 1.) / 2./ CompressRate) * CompressRate
    Z_Sample = np.ogrid[-m_:m_+1:CompressRate] + m
    PhantomFilted_ = PhantomFilted[:, :, Z_Sample]


    PhantomFilted_[torch.isnan(PhantomFilted_)] = param.threshold
    PhantomFilted_[PhantomFilted_ < param.threshold] = param.threshold

    param.PhantomSize_zcp = list(PhantomFilted_.size())

    return PhantomFilted_, param


def Func_ZCP_PSF(PSF, param):
    """
    Z向压缩PSF Z-direction compression PSF
    :param PSF: PSF
    :param param: 压缩参数 Compression parameters
    :return: 压缩后的PSF Compressed PSF
    """
    PhantomSize = param.PSFSize
    kernel_size = PhantomSize[4]
    CompressRate = param.CompressRate

    m = int((kernel_size - 1.) / 2.)
    m_ = np.floor((kernel_size - 1.) / 2./ CompressRate) * CompressRate
    Z_Sample = np.ogrid[-m_:m_+1:CompressRate]+m
    PSF_ = PSF[:, :,:, :,  Z_Sample]

    param.PSFSize_zcp = list(PSF_.size())

    return PSF_, param


def Func_DeZCP_ErrorBack(ErrorBack, param):
    """
    Z方向压缩后误差反传的解压过程 Decompression process of error back propagation after compression in Z direction
    :param ErrorBack: 误差 error
    :param param: parameters
    :return: 解压后的误差 Error after decompression
    """
    CompressRate = param.CompressRate
    ErrorBackSize = list(ErrorBack.size())

    ErrorBack = ErrorBack.reshape([1, 1, ErrorBackSize[0], ErrorBackSize[1], ErrorBackSize[2]])
    ErrorBack_ = F.interpolate(ErrorBack, size=None, scale_factor=(1, 1, CompressRate), mode='trilinear', align_corners=False)

    ErrorBack_ = ErrorBack_.squeeze()

    ErrorBack_Size = list(ErrorBack_.size())
    PhantomSize = param.PhantomSize

    edge = int((PhantomSize[2]-ErrorBack_Size[2]) / 2)

    ErrorBack__ = torch.zeros(tuple(PhantomSize), dtype=torch.float).to(param.device)

    if edge == 0:
        ErrorBack__ = ErrorBack_
    elif edge > 0:
        ErrorBack__[:, :, edge:-edge] = ErrorBack_
    else:
        ErrorBack__ = ErrorBack_[:, :, -edge:edge]

    return ErrorBack__, param


def Func_ZCP_param(param):



    return param


def RLReconstructionFourier3_SingleAngle(param, Phantom, PSF):
    """
    无像差的重建流程 Aberration-free reconstruction process
    :param param: 参数 parameters
    :param Phantom: 强度矩阵 intensity Matrix
    :param PSF: PSF
    :return: param, Phantom
    """
    if param.CompressRate == 1:
        Phantom_zcp = Phantom
        PSF_zcp = PSF
        param.PSFSize_zcp = param.PSFSize
        param.PhantomSize_zcp = param.PhantomSize
    else:
        CompressRate = param.CompressRate
        Phantom_zcp, param = Func_ZCP_Phantom(Phantom, param) # todo 这是整个运算的内存峰值，需要优化
        PSF_zcp, param = Func_ZCP_PSF(PSF, param)
        param = Func_ZCP_param(param)
        torch.cuda.empty_cache()


    Phantom_zcp = Phantom_zcp.to(param.device)

    ## loading meta parameter
    RotAngle = param.RotAngle
    i_RotAngle = param.i_RotAngle
    map = param.map
    PSFSize = param.PSFSize_zcp
    PhantomSize = param.PhantomSize_zcp

    ## Loading Projrction
    WignerPath = param.WignerPath

    ## Loading Projrction
    # data = io.loadmat(WignerPath + '//' + str(param.i_RotAngle).rjust(2, '0') + '_' + str(param.RotAngle).rjust(2, '0'))
    # data = io.loadmat(WignerPath + '/Projrction_' + str(param.i_RotAngle) + '_' + str(param.RotAngle))
    # data = io.loadmat(WignerPath + '/Projrction_' + str(0).rjust(2, '0') + '_' + str(24).rjust(2, '0'))
    # Projrction = data['Projrction']



    # addr = WignerPath + '/' + str(param.i_RotAngle).rjust(2, '0') + '_' + str(param.RotAngle).rjust(2, '0')
    addr = WignerPath + '/' + str(1) + '_' + str(1) + '_' + str(param.i_RotAngle +1)
    addr = WignerPath + '/' + str(1) + '_' + str(param.i_RotAngle +1)

    # addr = WignerPath + '/Projrction_' + str(param.i_RotAngle) + '_' + str(param.RotAngle)

    tif_Wigner = TIFF.open(addr + '.tif', mode='r')

    Image_stack = np.zeros([PhantomSize[0], PhantomSize[1], PSFSize[2]*PSFSize[3]])
    i = 0
    for image in tif_Wigner.iter_images():
        Image_stack[param.AddEdge[0]: PhantomSize[0]-param.AddEdge[0], param.AddEdge[1]: PhantomSize[1]-param.AddEdge[1], i] = image
        i = i + 1


    Image_stack = np.maximum(Image_stack - param.BackGround, 0)
    Image_stack = Image_stack.astype('float32')
    Image_stack = torch.from_numpy(Image_stack)
    Image_stack = Image_stack.to(param.device)

    ErrorBack_Iter = 1
    for i_Map in range(0, map.shape[0]):
        time_1 = time()

        i_ProjAngle = map[i_Map, 0] - 1
        j_ProjAngle = map[i_Map, 1] - 1
        param.ProjAngle = [i_ProjAngle, j_ProjAngle]
        ProjNum = i_ProjAngle * PSFSize[2] + j_ProjAngle

        PSF_SingleAngle = PSF_zcp[:, :, i_ProjAngle, j_ProjAngle, :].to(param.device)

        Image = Image_stack[:, :, ProjNum]

        ErrorBack = Func_FPBP.RL_Reconstruction_FPBP(param, Phantom_zcp, PSF_SingleAngle, Image)

        Phantom_zcp = (ErrorBack * param.StepRate + (1 - param.StepRate)) * Phantom_zcp

        ErrorBack_Iter = (ErrorBack * param.StepRate + (1 - param.StepRate)) * ErrorBack_Iter


        # threshold = param.threshold
        # Phantom_threshold = torch.zeros_like(Phantom) + threshold
        # Phantom = torch.where(Phantom < threshold, Phantom_threshold, Phantom)
        # Phantom = torch.where(torch.isnan(Phantom), Phantom_threshold, Phantom)



        # print(time() - time_1)

        # loss = criterion(Projrction_thisangle, Projrction_thisangle_fix)
        # Loss_list[epoch * RotAngle * map.shape[0] + map.shape[0] * i_RotAngle + i_Map] = loss.item()
        # io.savemat(SavePath + 'Loss', {'Loss': Loss_list})

        error_ = torch.mean(ErrorBack)
        sys.stdout.write("[Train] [Epoch {}/{}] [Rotation {}/{}] [Angle {}/{}] [loss:{:.5f}] time {:.3f}\n"
                         .format(param.epoch, param.EpochNum, param.i_RotAngle, param.RotAngle, i_Map, map.shape[0], error_,
                                 time() - time_1))
        # sys.stdout.flush()

    tif_Wigner.close()

    if param.CompressRate == 1:
        ErrorBack_Iter = ErrorBack_Iter
    else:
        ErrorBack_Iter, param = Func_DeZCP_ErrorBack(ErrorBack_Iter, param)

    Phantom = ErrorBack_Iter * Phantom
    # del ErrorBack_Iter
    # torch.cuda.empty_cache()

    Phantom[torch.isnan(Phantom)] = param.threshold
    Phantom[Phantom < param.threshold] = param.threshold

    # Phantom = Phantom.to(param_RLRecon.deviceOut)


    if param.Flag_AngleSave == True:
        SavePath = param.SavePath
        epoch = param.epoch
        i_RotAngle = param.i_RotAngle
        Phantom_numpy = np.squeeze(Phantom.cpu().detach().numpy())
        addr = SavePath + 'Phantom_' + str(epoch) + '_' + str(i_RotAngle) + '_' + str(i_Map)
        io.savemat(addr + '.mat', {'a': 1, 'Phantom_numpy': Phantom_numpy})
        tif = TIFF.open(addr + '.tif', mode='w')
        for i in range(0, Phantom_numpy.shape[2]):
            tif.write_image(np.squeeze(Phantom_numpy[:, :, i]))
        tif.close()


    return param, Phantom, ErrorBack_Iter


def SyncRLReconstructionFourier3_SingleAngle(param, Phantom, PSF):
    """
    各wigner同步化的无像差的重建流程 wigner Sync Aberration-free reconstruction process
    :param param: 参数 parameters
    :param Phantom: 强度矩阵 intensity Matrix
    :param PSF: PSF
    :return: param, Phantom
    """
    if param.CompressRate == 1:
        Phantom_zcp = Phantom
        PSF_zcp = PSF
        param.PSFSize_zcp = param.PSFSize
        param.PhantomSize_zcp = param.PhantomSize
    else:
        CompressRate = param.CompressRate
        Phantom_zcp, param = Func_ZCP_Phantom(Phantom, param) # todo 这是整个运算的内存峰值，需要优化
        PSF_zcp, param = Func_ZCP_PSF(PSF, param)
        param = Func_ZCP_param(param)
        torch.cuda.empty_cache()


    Phantom_zcp = Phantom_zcp.to(param.device)

    ## loading meta parameter
    RotAngle = param.RotAngle
    i_RotAngle = param.i_RotAngle
    map = param.map
    PSFSize = param.PSFSize_zcp
    PhantomSize = param.PhantomSize_zcp

    ## Loading Projrction
    WignerPath = param.WignerPath

    ## Loading Projrction
    # data = io.loadmat(WignerPath + '//' + str(param.i_RotAngle).rjust(2, '0') + '_' + str(param.RotAngle).rjust(2, '0'))
    # data = io.loadmat(WignerPath + '/Projrction_' + str(param.i_RotAngle) + '_' + str(param.RotAngle))
    # data = io.loadmat(WignerPath + '/Projrction_' + str(0).rjust(2, '0') + '_' + str(24).rjust(2, '0'))
    # Projrction = data['Projrction']



    # addr = WignerPath + '/' + str(param.i_RotAngle).rjust(2, '0') + '_' + str(param.RotAngle).rjust(2, '0')
    addr = WignerPath + '/' + str(1) + '_' + str(param.i_RotAngle +1)
    # addr = WignerPath + '/Projrction_' + str(param.i_RotAngle) + '_' + str(param.RotAngle)

    tif_Wigner = TIFF.open(addr + '.tif', mode='r')

    Image_stack = np.zeros([PhantomSize[0], PhantomSize[1], PSFSize[2]*PSFSize[3]])
    i = 0
    for image in tif_Wigner.iter_images():
        Image_stack[param.AddEdge[0]: PhantomSize[0]-param.AddEdge[0], param.AddEdge[1]: PhantomSize[1]-param.AddEdge[1], i] = image
        i = i + 1


    Image_stack = np.maximum(Image_stack - param.BackGround, 0)
    Image_stack = Image_stack.astype('float32')
    Image_stack = torch.from_numpy(Image_stack)
    Image_stack = Image_stack.to(param.device)

    ErrorBack_Iter = 1
    for i_Map in range(0, map.shape[0]):
        time_1 = time()

        i_ProjAngle = map[i_Map, 0] - 1
        j_ProjAngle = map[i_Map, 1] - 1
        param.ProjAngle = [i_ProjAngle, j_ProjAngle]
        ProjNum = i_ProjAngle * PSFSize[2] + j_ProjAngle

        PSF_SingleAngle = PSF_zcp[:, :, i_ProjAngle, j_ProjAngle, :].to(param.device)

        Image = Image_stack[:, :, ProjNum]

        ErrorBack = Func_FPBP.RL_Reconstruction_FPBP(param, Phantom_zcp, PSF_SingleAngle, Image)

        # Phantom_zcp = (ErrorBack * param.StepRate + (1 - param.StepRate)) * Phantom_zcp

        ErrorBack_Iter = (ErrorBack ** (1/map.shape[0])) * ErrorBack_Iter


        # threshold = param.threshold
        # Phantom_threshold = torch.zeros_like(Phantom) + threshold
        # Phantom = torch.where(Phantom < threshold, Phantom_threshold, Phantom)
        # Phantom = torch.where(torch.isnan(Phantom), Phantom_threshold, Phantom)



        # print(time() - time_1)

        # loss = criterion(Projrction_thisangle, Projrction_thisangle_fix)
        # Loss_list[epoch * RotAngle * map.shape[0] + map.shape[0] * i_RotAngle + i_Map] = loss.item()
        # io.savemat(SavePath + 'Loss', {'Loss': Loss_list})

        error_ = torch.mean(ErrorBack)
        sys.stdout.write("[Train] [Epoch {}/{}] [Rotation {}/{}] [Angle {}/{}] [loss:{:.5f}] time {:.3f}\n"
                         .format(param.epoch, param.EpochNum, param.i_RotAngle, param.RotAngle, i_Map, map.shape[0], error_,
                                 time() - time_1))
        # sys.stdout.flush()

    tif_Wigner.close()

    if param.CompressRate == 1:
        ErrorBack_Iter = ErrorBack_Iter
    else:
        ErrorBack_Iter, param = Func_DeZCP_ErrorBack(ErrorBack_Iter, param)

    Phantom = (ErrorBack_Iter * param.StepRate + (1 - param.StepRate)) * Phantom
    # del ErrorBack_Iter
    # torch.cuda.empty_cache()

    Phantom[torch.isnan(Phantom)] = param.threshold
    Phantom[Phantom < param.threshold] = param.threshold

    # Phantom = Phantom.to(param_RLRecon.deviceOut)


    if param.Flag_AngleSave == True:
        SavePath = param.SavePath
        epoch = param.epoch
        i_RotAngle = param.i_RotAngle
        Phantom_numpy = np.squeeze(Phantom.cpu().detach().numpy())
        addr = SavePath + 'Phantom_' + str(epoch) + '_' + str(i_RotAngle) + '_' + str(i_Map)
        io.savemat(addr + '.mat', {'a': 1, 'Phantom_numpy': Phantom_numpy})
        tif = TIFF.open(addr + '.tif', mode='w')
        for i in range(0, Phantom_numpy.shape[2]):
            tif.write_image(np.squeeze(Phantom_numpy[:, :, i]))
        tif.close()


    return param, Phantom, ErrorBack_Iter


def EdgeCut(Phantom, param):
    """
    编远切除
    :param Phantom: 强度矩阵 intensity Matrix
    :param param: 参数 parameters
    :return: 强度矩阵  Edge Cuted intensity Matrix
    """
    CutEdge = param.CutEdge
    Th = param.threshold
    PhantomSize = param.PhantomSize

    temp1 = int(CutEdge[0])
    temp2 = int(PhantomSize[0] - temp1-1)
    Phantom[0: temp1, :, :] = Th
    Phantom[temp2: , :, :] = Th

    temp1 = int(CutEdge[1])
    temp2 = int(PhantomSize[1] - temp1-1)
    Phantom[:, 0: temp1, :] = Th
    Phantom[:, temp2: , :] = Th

    temp1 = int(CutEdge[2])
    temp2 = int(PhantomSize[2] - temp1-1)
    Phantom[:, :, 0: temp1] = Th
    Phantom[:, :, temp2: ] = Th

    return Phantom


def RotationAberrationRLReconstructionFourier3_SingleAngle_ErrorOut(param_RLRecon, param_aberration, param_ShiftShiftMap, Phantom,
                                                           RefractionIndex, RayInit, PSF):
    """
    输出误差的基于Richardson–Lucy的单角度优化流程 Richardson–Lucy-based single-angle optimization process with Error as output
    :param param_RLRecon: Richardson–Lucy重建参数 Richardson–Lucy reconstruction parameters
    :param param_aberration: 正向像差参数 aberration parameter
    :param param_ShiftShiftMap: 反向像差参数 Reverse aberration parameters
    :param Phantom: 强度矩阵 intensity Matrix
    :param RefractionIndex: 折射率矩阵（本项目未使用）Refraction Index Matrix(Not used in this project)
    :param RayInit: 射线矩阵 wavefront Ray Matrix
    :param PSF: PSF
    :return:
    """
    torch.cuda.empty_cache()

    # Rotation
    Phantom = Phantom.to(param_RLRecon.device)
    param_RLRecon = Theta2TransMatrix(param_RLRecon)

    if param_RLRecon.thetaZ != 0:
        Phantom = Rotation_Z(Phantom, param_RLRecon)
    if param_RLRecon.thetaY != 0:
        Phantom = Rotation_Y(Phantom, param_RLRecon)
    if param_RLRecon.thetaX != 0:
        Phantom = Rotation_X(Phantom, param_RLRecon)

    Roted_Phantom = Phantom

    del Phantom
    torch.cuda.empty_cache()

    # Aberration
    if param_RLRecon.FlagAberration:
        Roted_Phantom = Roted_Phantom.to(param_aberration.device)
        RefractionIndex = RefractionIndex.to(param_aberration.device)
        model_Aberration = Modle_A_G.Model_Aberration(copy.copy(param_aberration)).to(param_aberration.device)
        model_Aberration.param.FlagRotate = True
        warped_Roted_Phantom, pathsmesh_interp = model_Aberration.forward(Roted_Phantom, RefractionIndex, RayInit)
        warped_Roted_Phantom = torch.squeeze(warped_Roted_Phantom)

        del Roted_Phantom
        torch.cuda.empty_cache()

    else:
        warped_Roted_Phantom = Roted_Phantom
        del Roted_Phantom
        torch.cuda.empty_cache()




    # Reconstruction
    if param_RLRecon.WignerSync:
        [param_RLRecon, warped_Roted_Phantom, warped_Roted_ErrorBack] = SyncRLReconstructionFourier3_SingleAngle(param_RLRecon, warped_Roted_Phantom, PSF) # 16530mb
    else:
        [param_RLRecon, warped_Roted_Phantom, warped_Roted_ErrorBack] = RLReconstructionFourier3_SingleAngle(param_RLRecon, warped_Roted_Phantom, PSF) # 16530mb

    del warped_Roted_Phantom
    torch.cuda.empty_cache()



    # Edge Cut
    warped_Roted_ErrorBack = EdgeCut(warped_Roted_ErrorBack, param_RLRecon)

    # Gfilter
    if (param_RLRecon.GHWFMX > 0.01) | (param_RLRecon.GHWFMY > 0.01) | (param_RLRecon.GHWFMZ > 0.01):
        warped_Roted_ErrorBack, _ = Func_Filter_Phantom(warped_Roted_ErrorBack, param_RLRecon)

    # deAberration
    if param_RLRecon.FlagAberration:
        pathsmesh_interp_unwarp = Func_DeAberration.Func_BackwardEstimation(pathsmesh_interp, param_ShiftShiftMap) #17691mb
        del pathsmesh_interp, param_ShiftShiftMap
        torch.cuda.empty_cache()  #  5500mb

        PhantomSize = param_RLRecon.PhantomSize
        warped_Roted_ErrorBack_ = warped_Roted_ErrorBack.reshape(1, 1, PhantomSize[0], PhantomSize[1], PhantomSize[2])
        Roted_ErrorBack = F.grid_sample(warped_Roted_ErrorBack_, pathsmesh_interp_unwarp,
                                               mode='bilinear', padding_mode='zeros', align_corners=True)
        del warped_Roted_ErrorBack_, pathsmesh_interp_unwarp
        torch.cuda.empty_cache()  #  5500mb
        Roted_ErrorBack = Roted_ErrorBack.squeeze()
    else:
        Roted_ErrorBack = warped_Roted_ErrorBack

    # deRotation
    Roted_ErrorBack_forAVG = Roted_ErrorBack
    Roted_ErrorBack_forAVG = torch.from_numpy(param_RLRecon.AvgMusk).to(param_RLRecon.device) * Roted_ErrorBack_forAVG

    param_RLRecon.thetaX = -param_RLRecon.thetaX
    param_RLRecon.thetaY = -param_RLRecon.thetaY
    param_RLRecon.thetaZ = -param_RLRecon.thetaZ
    param_RLRecon = Theta2TransMatrix(param_RLRecon)

    warped_ErrorBack = warped_Roted_ErrorBack # 用来指导波前迭代重建
    ErrorBack = Roted_ErrorBack
    ErrorBack_forAVG = Roted_ErrorBack_forAVG

    if param_RLRecon.thetaX != 0:
        warped_ErrorBack = Rotation_X(warped_ErrorBack, param_RLRecon)
        ErrorBack = Rotation_X(ErrorBack, param_RLRecon)
        ErrorBack_forAVG = Rotation_X(ErrorBack_forAVG, param_RLRecon)
    if param_RLRecon.thetaY != 0:
        warped_ErrorBack = Rotation_X(warped_ErrorBack, param_RLRecon)
        ErrorBack = Rotation_Y(ErrorBack, param_RLRecon)
        ErrorBack_forAVG = Rotation_Y(ErrorBack_forAVG, param_RLRecon)
    if param_RLRecon.thetaZ != 0:
        warped_ErrorBack = Rotation_X(warped_ErrorBack, param_RLRecon)
        ErrorBack = Rotation_Z(ErrorBack, param_RLRecon)
        ErrorBack_forAVG = Rotation_Z(ErrorBack_forAVG, param_RLRecon)


    torch.cuda.empty_cache() # 2235 mb

    return ErrorBack, ErrorBack_forAVG, warped_ErrorBack
    # return warped_Phantom


def RotationAberrationSyncErrorRLReconstructionFourier3(param_RLRecon, param_ShiftShiftMap, Phantom,
                                               RefractionIndex, RayInitList, PSF):
    """
    基于Richardson–Lucy的多角度全局优化流程 Richardson–Lucy-based multi-angle global optimization process
    :param param_RLRecon: Richardson–Lucy重建参数 Richardson–Lucy reconstruction parameters
    :param param_ShiftShiftMap: 反向像差参数 Reverse aberration parameters
    :param Phantom: 强度矩阵 intensity Matrix
    :param RefractionIndex: 折射率矩阵（本项目未使用）Refraction Index Matrix(Not used in this project)
    :param RayInitList: 射线矩阵 wavefront Ray Matrix
    :param PSF: PSF
    :return:
    """

    param_aberration = InitAberParam.Func_InitializeAberrationParameter(param_RLRecon)

    PhantomSize = param_RLRecon.PhantomSize
    EpochNum = param_RLRecon.EpochNum
    RotAngle = param_RLRecon.RotAngle
    RotAngleSkip = param_RLRecon.RotAngleSkip

    Phantom_AVG = Phantom.to(param_RLRecon.device)
    RayInitList = RayInitList.to(param_RLRecon.device)

    Phantom_List = torch.zeros(int(RotAngle/RotAngleSkip), PhantomSize[0], PhantomSize[1], PhantomSize[2])
    warped_Phantom_List = torch.zeros(int(RotAngle/RotAngleSkip), PhantomSize[0], PhantomSize[1], PhantomSize[2])

    ErrorBack_IterAVG = 1
    ErrorBack_Iter = 1
    for epoch in range(0, EpochNum):
        param_RLRecon.epoch = epoch
        Phantom_ = Phantom_AVG * param_RLRecon.MagnifyRate

        # for i in range(0, RotAngle):
        for i_RotAngle in range(0, RotAngle, RotAngleSkip):

            param_RLRecon.i_RotAngle = i_RotAngle
            param_ShiftShiftMap.i_RotAngle = i_RotAngle

            # 所有角度都是正方向右手螺旋 旋转顺序ZYX
            param_RLRecon.thetaZ = i_RotAngle * param_RLRecon.dthetaZ
            param_RLRecon.thetaY = i_RotAngle * param_RLRecon.dthetaY
            param_RLRecon.thetaX = i_RotAngle * param_RLRecon.dthetaX

            param_aberration.thetaZ = i_RotAngle * param_aberration.dthetaZ
            param_aberration.thetaY = i_RotAngle * param_aberration.dthetaY
            param_aberration.thetaX = i_RotAngle * param_aberration.dthetaX

            # Phantom = Func_IntensityRecon.RotationAberrationRLReconstructionFourier3_SingleAngle(
            #     param_RLRecon, Phantom, PSF)
            ErrorBack, ErrorBack_forAVG, warped_ErrorBack = RotationAberrationRLReconstructionFourier3_SingleAngle_ErrorOut(
                param_RLRecon, param_aberration, param_ShiftShiftMap, Phantom_,
                RefractionIndex, RayInitList[i_RotAngle, :, :, :], PSF)

            torch.cuda.empty_cache()
            warped_Phantom_List[int(i_RotAngle/RotAngleSkip), :, :, :] = Phantom.cpu().detach()
            Phantom_List[int(i_RotAngle/RotAngleSkip), :, :, :] = Phantom.cpu().detach()
            ErrorBack_IterAVG = (ErrorBack_forAVG ** (RotAngleSkip/RotAngle)) * ErrorBack_IterAVG
            ErrorBack_Iter = (ErrorBack ** (RotAngleSkip/RotAngle)) * ErrorBack_Iter

            torch.cuda.empty_cache()

        Phantom_forAVG = ErrorBack_IterAVG * Phantom_
        Phantom = ErrorBack_Iter * Phantom_

        # Save

        if param_RLRecon.FlagTestSave:
            addr = param_RLRecon.SavePathTest + 'Phantom_' + str(param_RLRecon.BigEpoch) + '_' + str(epoch)
            TiffStackSave(Phantom, addr)

            addr = param_RLRecon.SavePathTest + 'Phantom_forAVG_' + str(param_RLRecon.BigEpoch) + '_' + str(epoch)
            TiffStackSave(Phantom_forAVG, addr)

        addr = param_RLRecon.SavePath + 'Phantom_AVG_' + str(epoch)
        TiffStackSave(Phantom_forAVG, addr)

        addr = param_RLRecon.SavePath + 'Phantom_' + str(epoch)
        TiffStackSave(Phantom, addr)


    return Phantom_List, Phantom_forAVG, warped_Phantom_List
