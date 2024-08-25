
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


def RL_Reconstruction_FPBP(param, Phantom, PSF_SingleAngle, Image):
    """
    Richardson–Lucy重建算法正向传播与反向传播后获得
    Error Error obtained after forward and back propagation of Richardson–Lucy reconstruction algorithm
    :param param: 参数 parameter
    :param Phantom: 强度矩阵 intensity Matrix
    :param PSF_SingleAngle: 单角度的PSF PSF of a single angle
    :param Image: Ground Truth
    :return: ErrorBack 误差
    """
    PhantomSize = param.PhantomSize_zcp
    PSFSize = param.PSFSize_zcp


    PSF_SingleAngle = PSF_SingleAngle.to(param.device)
    # plt.imshow(np.squeeze(Phantom[:, :, 100].cpu().detach().numpy()))
    pd_temp_1 = int((PhantomSize[0] - PSFSize[0]) / 2)
    pd_temp_2 = int((PhantomSize[1] - PSFSize[1]) / 2)
    pd_temp = (0, 0, pd_temp_2, pd_temp_2, pd_temp_1, pd_temp_1)
    pd_temp = [0, 0, pd_temp_2, pd_temp_2, pd_temp_1, pd_temp_1]

    PSF_SingleAngle_enlarge = F.pad(PSF_SingleAngle, pd_temp, "constant", 0)

    temp = sum(sum(sum(PSF_SingleAngle_enlarge, 0), 0), 0)
    PSF_SingleAngle_enlarge = PSF_SingleAngle_enlarge/temp

    PSF_SingleAngle_enlarge_1 = torch.flip(PSF_SingleAngle_enlarge, [2])
    PSF_SingleAngle_enlarge_2 = torch.rot90(PSF_SingleAngle_enlarge, 2, [0, 1])



    PSF_SingleAngle_enlarge_1_fft = torch.fft.fftn(PSF_SingleAngle_enlarge_1)
    PSF_SingleAngle_enlarge_2_fft = torch.fft.fftn(PSF_SingleAngle_enlarge_2)

    del PSF_SingleAngle, PSF_SingleAngle_enlarge, PSF_SingleAngle_enlarge_1, PSF_SingleAngle_enlarge_2

    Phantom_fft = torch.fft.fftn(Phantom)

    conv_fft = PSF_SingleAngle_enlarge_1_fft * Phantom_fft
    del PSF_SingleAngle_enlarge_1_fft
    conv_2d_fft = torch.sum(conv_fft, 2) / PhantomSize[2]

    ############# 这里可能需要转换维度
    conv2_2d5_fft = conv_2d_fft.repeat(PhantomSize[2], 1, 1)
    conv2_2d5_fft = conv2_2d5_fft.permute([1, 2, 0])

    PhantomBack_fft = PSF_SingleAngle_enlarge_2_fft * conv2_2d5_fft

    PhantomBack = torch.fft.ifftn(PhantomBack_fft, s=PhantomSize[0:3])
    PhantomBack = torch.real(PhantomBack)

    shift_temp = int(1)
    PhantomBack = torch.roll(PhantomBack, shifts=(shift_temp, shift_temp), dims=(0, 1))
    # shift_temp = int((525 + 1) / 2)
    # PhantomBack__ = torch.roll(PhantomBack_, shifts=(shift_temp), dims=(2))

    PhantomBack[torch.isnan(PhantomBack)] = param.threshold
    PhantomBack[PhantomBack < param.threshold] = param.threshold

    del PhantomBack_fft, conv2_2d5_fft

    # Addr = 'PhantomBack'
    # PhantomBack_numpy = np.squeeze(PhantomBack.cpu().detach().numpy())
    # io.savemat(Addr + '.mat', {'a': 1, 'PhantomBack': PhantomBack_numpy})
    # tif = TIFF.open(Addr + '.tif', mode='w')
    # for i in range(0, PhantomBack_numpy.shape[2]):
    #     tif.write_image(np.squeeze(PhantomBack_numpy[:, :, i]))
    # tif.close()




    Image_fft = torch.fft.fft2(Image)
    ############# 这里可能需要转换维度
    Image_fft = Image_fft.repeat(PhantomSize[2], 1, 1)
    Image_fft = Image_fft.permute([1, 2, 0])

    ImageBack_fft = PSF_SingleAngle_enlarge_2_fft * Image_fft
    del Image_fft, PSF_SingleAngle_enlarge_2_fft

    ImageBack = torch.fft.ifftn(ImageBack_fft, s=PhantomSize[0:3])
    ImageBack = torch.real(ImageBack)

    shift_temp = int((PhantomSize[1] + 1) / 2)
    ImageBack = torch.roll(ImageBack, shifts=(shift_temp, shift_temp), dims=(0, 1))

    ImageBack[torch.isnan(ImageBack)] = param.threshold
    ImageBack[ImageBack < param.threshold] = param.threshold

    del ImageBack_fft

    # Addr = 'ImageBack'
    # ImageBack_numpy = np.squeeze(ImageBack.cpu().detach().numpy())
    # io.savemat(Addr + '.mat', {'a': 1, 'ImageBack': ImageBack_numpy})
    # tif = TIFF.open(Addr + '.tif', mode='w')
    # for i in range(0, ImageBack_numpy.shape[2]):
    #     tif.write_image(np.squeeze(ImageBack_numpy[:, :, i]))
    # tif.close()




    ErrorBack = ImageBack / PhantomBack
    # ImageBack[torch.isnan(ImageBack)] = param.threshold
    # ImageBack[ImageBack < param.threshold] = param.threshold

    return ErrorBack
