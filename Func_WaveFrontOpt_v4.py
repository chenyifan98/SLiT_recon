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

from tensorboardX import SummaryWriter

import Func_zernike_v1

import ModelAberration_Rot_v4 as Modle_A_G
import Func_InitializeAberrationParameter_v2 as InitAberParam
from Func_TiffStackSave import TiffStackSave as TiffStackSave
from Func_TiffStackSave import TiffStackLoad as TiffStackLoad

class parameter:
    def __init__(self):
        # device
        self.device = []


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


def DownSample3D(Phantom, param):
    """
    强度矩阵下采样 Intensity Matrix Downsampling
    :param Phantom: Intensity Matrix
    :param param: parameters
    :return:
    """
    Phantom = Phantom.to(param.device)

    PhantomSize = param.PhantomSize
    kernel_size = PhantomSize[2]

    CompressRate = param.DownSampleRate
    kernel_size = CompressRate * 5
    std = 2 * CompressRate / 2.355
    m = int((kernel_size - 1.) / 2.)
    x = np.ogrid[-m:m + 1]
    y = np.ogrid[-m:m + 1]
    z = np.ogrid[-m:m + 1]
    x, y, z = np.meshgrid(x, y, z)

    h = np.exp(-(x * x + y * y + z * z) / (2. * std * std))


    sumh = h.sum()
    h /= sumh

    Filter = torch.zeros(PhantomSize, dtype=torch.float32)
    temp1 = ((np.array(PhantomSize)-1)/2 - (kernel_size-1)/2).astype('int16')
    temp2 = ((np.array(PhantomSize)-1)/2 + (kernel_size-1)/2 + 1).astype('int16')
    Filter[temp1[0]:temp2[0], temp1[1]:temp2[1], temp1[2]:temp2[2]] = torch.from_numpy(h)
    Filter = Filter.to(param.device)



    Filter_fft = torch.fft.fftn(Filter)
    Phantom_fft = torch.fft.fftn(Phantom)
    del Filter, Phantom
    torch.cuda.empty_cache()

    conv_fft = Filter_fft * Phantom_fft
    del Filter_fft, Phantom_fft
    torch.cuda.empty_cache()


    PhantomFilted = torch.fft.ifftn(conv_fft, s=PhantomSize[0:3])
    PhantomFilted = torch.real(PhantomFilted)
    del conv_fft
    torch.cuda.empty_cache()

    shift_temp = ((np.array(PhantomSize) + 1) / 2).astype('int16')

    # Addr = 'PhantomFilted'
    # PhantomFilted_numpy = np.squeeze(PhantomFilted.cpu().detach().numpy())
    # io.savemat(Addr + '.mat', {'a': 1, 'ImageBack': PhantomFilted_numpy})
    # tif = TIFF.open(Addr + '.tif', mode='w')
    # for i in range(0, PhantomFilted_numpy.shape[2]):
    #     tif.write_image(np.squeeze(PhantomFilted_numpy[:, :, i]))
    # tif.close()

    PhantomFilted = torch.roll(PhantomFilted, shifts=(shift_temp[0], shift_temp[1], shift_temp[2]), dims=(0, 1, 2))


    # Downsample
    rate = param.DownSampleRate
    PhantomFilted = PhantomFilted.reshape([1,1,PhantomSize[0],PhantomSize[1],PhantomSize[2]])
    PhantomFilted = torch.nn.functional.interpolate(PhantomFilted, size=None, scale_factor=1/rate, mode='trilinear', align_corners=False, recompute_scale_factor=True)
    PhantomFilted = PhantomFilted.squeeze()

    return PhantomFilted


def PhantomNormalizeParam_std(Phantom, param):
    """
    强度矩阵正则化 Intensity Matrix Regularization
    :param Phantom:
    :param param:
    :return:
    """
    std = Phantom.std()
    mean = Phantom.mean()

    param.PhantomNormalizeMax = float(mean + std * 3)
    param.PhantomNormalizeMin = float(mean - std * 0)

    return param


def PhantomMinMaxNormalize(Phantom, param):
    """
    强度矩阵极值的正则化 Regularization of extreme values of the intensity matrix
    :param Phantom:
    :param param:
    :return:
    """
    Phantom = (Phantom - param.PhantomNormalizeMin)/(param.PhantomNormalizeMax - param.PhantomNormalizeMin)
    return Phantom


def RayInitOptimization_SingleIteration(param_RayInitOpt, PhantomAvg, Phantom_List, RefractionIndex, RayInitZernikeList):
    """
    射线矩阵(波前)的单次迭代优化 Single-iteration optimization of the ray matrix (wavefront)
    :param param_RayInitOpt: 射线的优化参数 Optimization parameters of rays
    :param PhantomAvg: 不同Viewpoint的强度矩阵均值 The mean value of the intensity matrix at different viewpoints
    :param Phantom_List: 不同Viewpoint的强度矩阵list intensity matrix list of different Viewpoints
    :param RefractionIndex: 折射率矩阵（本项目未使用）Refractive index matrix (not used in this project)
    :param RayInitZernikeList: 射线矩阵 Ray Matrix
    :return: param_RayInitOpt, RayInitZernikeList, RayInitList
    """
    BigEpoch = param_RayInitOpt.BigEpoch
    param_aberration = InitAberParam.Func_InitializeAberrationParameter(param_RayInitOpt)
    param_aberration.Num_RaySample = 2

    ## loading meta parameter
    RotAngle = param_RayInitOpt.RotAngle

    ## creat model
    # RefractionIndex
    A = RefractionIndex[:, :, :, 0]  # [y, x, z]
    sig = RefractionIndex[:, :, :, 1]  # [y, x, z]
    A = A.to(param_aberration.device)
    sig = sig.to(param_aberration.device)

    # RayInitList
    RayInitZernikeList = RayInitZernikeList.to(param_aberration.device)
    ZernikeItem = param_RayInitOpt.ZernikeItem.to(param_aberration.device)
    RayInitList0 = param_RayInitOpt.RayInitList0.to(param_aberration.device)

    # Phantom
    PhantomAvg = PhantomAvg.to(param_aberration.device)
    PhantomAvg = DownSample3D(PhantomAvg, param_RayInitOpt)
    PhantomSize = (list(PhantomAvg.size()))
    param_aberration.PhantomSize = PhantomSize

    param_RayInitOpt = PhantomNormalizeParam_std(PhantomAvg, param_RayInitOpt)
    PhantomAvg = PhantomMinMaxNormalize(PhantomAvg, param_RayInitOpt)

    Phantom_List_ = torch.zeros([RotAngle, PhantomSize[0], PhantomSize[1], PhantomSize[2]], dtype=torch.float32)
    for i in range(0, RotAngle):
        Phantom_List_[i, :, :, :] = DownSample3D(Phantom_List[i, :, :, :], param_RayInitOpt)  # todo 内存峰值需要解决
        Phantom_List_[i, :, :, :] = PhantomMinMaxNormalize(Phantom_List_[i, :, :, :], param_RayInitOpt)

    Phantom_List = Phantom_List_

    ## creat model
    model_Aberration = Modle_A_G.Model_Aberration(copy.copy(param_aberration)).to(param_aberration.device)

    ## Set Optimization
    criterion = torch.nn.MSELoss().to(param_aberration.device)
    criterion_L2norm = torch.nn.MSELoss().to(param_aberration.device)
    criterion_DfL1norm = torch.nn.L1Loss().to(param_aberration.device)
    criterion_DfL2norm = torch.nn.MSELoss().to(param_aberration.device)

    OptMusk = torch.tensor(param_RayInitOpt.OptMusk, device=param_RayInitOpt.device).float()

    ## Train
    writer = SummaryWriter(param_RayInitOpt.SavePath)
    RayInitList = RayInitList0 * 0
    for i_RotAngle in range(0, RotAngle):

        RefractionIndex = torch.stack([A, sig], 3)

        # Rotation

        param_aberration.thetaZ = i_RotAngle * param_aberration.dthetaZ
        param_aberration.thetaY = i_RotAngle * param_aberration.dthetaY
        param_aberration.thetaX = i_RotAngle * param_aberration.dthetaX
        param_aberration = Theta2TransMatrix(param_aberration)

        model_Aberration.param.thetaZ = i_RotAngle * model_Aberration.param.dthetaZ
        model_Aberration.param.thetaY = i_RotAngle * model_Aberration.param.dthetaY
        model_Aberration.param.thetaX = i_RotAngle * model_Aberration.param.dthetaX

        ## PhantomAvg
        PhantomAvg_rot = PhantomAvg
        if param_aberration.thetaZ != 0:
            PhantomAvg_rot = Rotation_Z(PhantomAvg_rot, param_aberration)
        if param_aberration.thetaY != 0:
            PhantomAvg_rot = Rotation_Y(PhantomAvg_rot, param_aberration)
        if param_aberration.thetaX != 0:
            PhantomAvg_rot = Rotation_X(PhantomAvg_rot, param_aberration)

        # warped_Phantom_fix
        warped_Phantom_fix = Phantom_List[i_RotAngle, :, :, :].to(param_aberration.device)
        if param_aberration.thetaZ != 0:
            warped_Phantom_fix = Rotation_Z(warped_Phantom_fix, param_aberration)
        if param_aberration.thetaY != 0:
            warped_Phantom_fix = Rotation_Y(warped_Phantom_fix, param_aberration)
        if param_aberration.thetaX != 0:
            warped_Phantom_fix = Rotation_X(warped_Phantom_fix, param_aberration)

        ## Set Optimization Parameter
        lr = param_RayInitOpt.lr
        # RayInitZernike = RayInitZernikeList[i_RotAngle, :, :].squeeze()
        RayInitZernike = RayInitZernikeList[i_RotAngle, :, :]
        RayInitZernike = torch.nn.Parameter(RayInitZernike, requires_grad=True)
        optimizer = torch.optim.Adam([RayInitZernike], lr=lr)
        stop_epoch = param_RayInitOpt.EpochNum
        Loss_list = np.zeros(stop_epoch)
        ZernikeRank = RayInitZernike.shape[0]

        # os.popen('tensorboard --logdir ' + param_RayInitOpt.SavePath)

        for epoch in range(0, stop_epoch):
            time_1 = time()

            optimizer.zero_grad()

            # Zernike2WaveFront
            dRayInit = torch.zeros([param_RayInitOpt.Num_Ray, param_RayInitOpt.Num_Ray, 5], device=param_RayInitOpt.device).float()
            for i in range(0, 5):
                dRayInit[:, :, i] = torch.sum(ZernikeItem * RayInitZernike[:, i] * OptMusk[i], 2)
            RayInit = dRayInit + RayInitList0[i_RotAngle, : ,:, :]

            # forward
            warped_Phantom, _ = model_Aberration.forward(PhantomAvg_rot, RefractionIndex, RayInit)
            warped_Phantom = torch.squeeze(warped_Phantom)

            # Loss
            loss = criterion(warped_Phantom, warped_Phantom_fix)
            Loss_list[epoch] = loss.item()
            loss.backward(retain_graph=True)  # 背向传播

            sys.stdout.write("[Train] [Epoch {}/{}] [Rotation {}/{}] [loss:{:.8f}] time {:.3f}\n"
                             .format(epoch + 1, stop_epoch, i_RotAngle, RotAngle, loss.item(), time() - time_1))
            sys.stdout.flush()

            # step
            optimizer.step()  # 优化器进行更新
            torch.cuda.empty_cache()

            # Save Tensorboard
            writer.add_scalar('data_rot'+str(i_RotAngle)+'/error', loss.item(), epoch + stop_epoch * (BigEpoch-1))

            for i_rank in range(0, ZernikeRank):
                writer.add_scalars('data_rot' + str(i_RotAngle) + '/zernike_x',
                                   {'Z_'+ str(i_rank): RayInitZernike[i_rank, 0]},
                                   epoch + stop_epoch * (BigEpoch - 1))
                writer.add_scalars('data_rot' + str(i_RotAngle) + '/zernike_dx',
                                   {'Z_'+ str(i_rank): RayInitZernike[i_rank, 1]},
                                   epoch + stop_epoch * (BigEpoch - 1))
                writer.add_scalars('data_rot' + str(i_RotAngle) + '/zernike_y',
                                   {'Z_'+ str(i_rank): RayInitZernike[i_rank, 2]},
                                   epoch + stop_epoch * (BigEpoch - 1))
                writer.add_scalars('data_rot' + str(i_RotAngle) + '/zernike_dy',
                                   {'Z_'+ str(i_rank): RayInitZernike[i_rank, 3]},
                                   epoch + stop_epoch * (BigEpoch - 1))
                writer.add_scalars('data_rot' + str(i_RotAngle) + '/zernike_z',
                                   {'Z_'+ str(i_rank): RayInitZernike[i_rank, 4]},
                                   epoch + stop_epoch * (BigEpoch - 1))


            # writer.add_scalar('data_rot'+str(i_RotAngle)+'/error', loss.item(), epoch + stop_epoch * (BigEpoch-1))
            # writer.add_scalars('data_rot'+str(i_RotAngle)+'/zernike_x', {'Z00': RayInitZernike[0, 0],
            #                                       'Z1-1': RayInitZernike[1, 0],
            #                                       'Z11': RayInitZernike[2, 0]}, epoch + stop_epoch * (BigEpoch-1))
            #
            # writer.add_scalars('data_rot'+str(i_RotAngle)+'/zernike_dx', {'Z00': RayInitZernike[0, 1],
            #                                        'Z1-1': RayInitZernike[1, 1],
            #                                        'Z11': RayInitZernike[2, 1]}, epoch + stop_epoch * (BigEpoch-1))
            #
            # writer.add_scalars('data_rot'+str(i_RotAngle)+'/zernike_y', {'Z00': RayInitZernike[0, 2],
            #                                       'Z1-1': RayInitZernike[1, 2],
            #                                       'Z11': RayInitZernike[2, 2]}, epoch + stop_epoch * (BigEpoch-1))
            #
            # writer.add_scalars('data_rot'+str(i_RotAngle)+'/zernike_dy', {'Z00': RayInitZernike[0, 3],
            #                                        'Z1-1': RayInitZernike[1, 3],
            #                                        'Z11': RayInitZernike[2, 3]}, epoch + stop_epoch * (BigEpoch-1))

            # Save
            SavePath = param_RayInitOpt.SavePath
            io.savemat(SavePath + 'Loss_' + str(param_RayInitOpt.BigEpoch) + '_' + str(i_RotAngle), {'Loss': Loss_list})
            if epoch % 100 == 0:
                addr = SavePath + 'dRayInit_' + str(param_RayInitOpt.BigEpoch) + '_' + str(i_RotAngle)
                TiffStackSave(dRayInit, addr)

            # Test Save
            SavePath = param_RayInitOpt.SavePathTest
            if param_RayInitOpt.FlagTestSave:
                if epoch % 100 == 0:
                    addr = SavePath + 'warped_Phantom_' + str(param_RayInitOpt.BigEpoch) + '_' + str(
                        i_RotAngle) + '_' + str(epoch)
                    TiffStackSave(warped_Phantom, addr)

                    addr = SavePath + 'warped_Phantom_fix_' + str(param_RayInitOpt.BigEpoch) + '_' + str(
                        i_RotAngle) + '_' + str(epoch)
                    TiffStackSave(warped_Phantom_fix, addr)

        RayInitZernikeList[i_RotAngle, :, :] = RayInitZernike.detach()
        RayInitList[i_RotAngle, :, :, :] = RayInit.detach()

    writer.close()

    return param_RayInitOpt, RayInitZernikeList, RayInitList
