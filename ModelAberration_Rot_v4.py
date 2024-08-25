
import numpy as np
import os
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math
import scipy.io as io
import torch
import torch.nn.functional as F
import copy



# import torchvision.models as models

def indexdist(x_, y_, z_, RefractionIndex, parameter):
    """
    计算该位置折射率矩阵（与本项目无关）
    Calculate the refractive index matrix at this position (not relevant to this project)
    :param x_:
    :param y_:
    :param z_:
    :param RefractionIndex:
    :param parameter:
    :return:
    """
    # xyz = np.stack([x_, y_, z_], 0)
    # xyzp = xyz
    # X = xyzp[0]
    # Y = xyzp[1]
    # Z = xyzp[2]
    RK4Range = parameter.RK4Range
    Num_Ray2 = parameter.Num_Ray2
    Num_RIGuassKernel = parameter.Num_RIGuassKernel
    temp_0 = parameter.temp_0
    temp_1 = parameter.temp_1

    X = x_.reshape([Num_Ray2, 1])
    Y = y_.reshape([Num_Ray2, 1])
    Z = z_.reshape([Num_Ray2, 1])


    # 获取周边范围矩阵
    Xround = torch.round(X * Num_RIGuassKernel)
    Yround = torch.round(Y * Num_RIGuassKernel)
    Zround = torch.round(Z * Num_RIGuassKernel)

    # temp = torch.ones([Num_Ray2, RK4Range ** 3])
    # temp_0 = temp * 0
    # temp_1 = temp * (Num_RIGuassKernel-1)



    Xneigh = torch.max(Xround + parameter.Xr, temp_0)
    Xneigh = torch.min(Xneigh, temp_1)
    Yneigh = torch.max(Yround + parameter.Yr, temp_0)
    Yneigh = torch.min(Yneigh, temp_1)
    Zneigh = torch.max(Zround + parameter.Zr, temp_0)
    Zneigh = torch.min(Zneigh, temp_1)

    # 获取相关点坐标编号
    # lininds = parameter.Num_RIGuassKernel * Zneigh + Xneigh  # convert to linear indices
    lininds = (Num_RIGuassKernel ** 2) * Yneigh + Num_RIGuassKernel * Xneigh + Zneigh
    lininds = lininds.long()

    # 提取位置 强度 方差
    Xg = parameter.Xc[lininds]
    Yg = parameter.Yc[lininds]
    Zg = parameter.Zc[lininds]
    A_ = RefractionIndex[lininds, 0]
    sig_ = RefractionIndex[lininds, 1]

    # add something close to machine eps for float32 to avoid divide by 0:
    # intermediate, will be used several times:
    n_intermed = torch.exp(-((Xg - X) ** 2 + (Yg - Y) ** 2 + (Zg - Z) ** 2) * .5 / (sig_ ** 2)) + 2e-7
    # normalize the sum (cf, nadaraya-watson):
    norm = torch.sum(n_intermed, axis=1)
    n_unnorm = n_intermed * A_  # unnormalized, used to calculate gradients

    nx = n_unnorm * (Xg - X) / sig_ ** 2
    ny = n_unnorm * (Yg - Y) / sig_ ** 2
    nz = n_unnorm * (Zg - Z) / sig_ ** 2
    nx = torch.sum(nx, axis=1)
    ny = torch.sum(ny, axis=1)
    nz = torch.sum(nz, axis=1)

    norm_deriv_x = n_intermed * (Xg - X) / sig_ ** 2
    norm_deriv_y = n_intermed * (Yg - Y) / sig_ ** 2
    norm_deriv_z = n_intermed * (Zg - Z) / sig_ ** 2
    norm_deriv_x = torch.sum(norm_deriv_x, axis=1)
    norm_deriv_y = torch.sum(norm_deriv_y, axis=1)
    norm_deriv_z = torch.sum(norm_deriv_z, axis=1)

    # now done with this, so collapse:
    n_unnorm = torch.sum(n_unnorm, axis=1)

    # now, compute the index distribution:
    n = n_unnorm / norm

    # ...and the spatial gradients (quotient rule)
    nx = (norm * nx - n_unnorm * norm_deriv_x) / norm ** 2
    ny = (norm * ny - n_unnorm * norm_deriv_y) / norm ** 2
    nz = (norm * nz - n_unnorm * norm_deriv_z) / norm ** 2

    # grad_ = np.stack([nx, ny, nz])
    # grad = grad_
    # grad = tf.matmul(grad_, self_.rotmats)

    # return n, grad[0], grad[1], grad[2]
    return n, nx, ny, nz


def rayeq_opl(z0, xy0, RefractionIndex, param):
    """
    光线追踪的高斯拟合步进（与本项目无关）
    Gaussian fitting stepping for ray tracing (not relevant to this project)
    :param z0:
    :param xy0:
    :param RefractionIndex:
    :param param:
    :return:
    """
    # (deriv, n) = rayeq_opl(z0, x0)


    # coordinate Rot
    z0 = z0
    xy0_x = xy0[:, 0]
    xy0_y = xy0[:, 2]
    temp_ones = torch.ones([param.Num_Ray ** 2]).float().to(param.device)
    temp_123 = torch.stack([xy0_y, xy0_x, z0, temp_ones], axis=1)
    temp_123_rot = temp_123.mm(param.TransMatrix_3DCrdRot)# todo 这里的旋转都乘反了

    ##
    (n, dndx_, dndy_, dndz_) = indexdist(temp_123_rot[:, 1], temp_123_rot[:, 0], temp_123_rot[:, 2], RefractionIndex, param)

    # derivitive Rot
    temp_dnd123= torch.stack([dndy_, dndx_, dndz_, temp_ones], axis=1)
    temp_dnd123_rot = temp_dnd123.mm(param.TransMatrix_3DDrvRot) # todo 这里的旋转都乘反了

    dndx = temp_dnd123_rot[:, 1]
    dndz = temp_dnd123_rot[:, 2]
    dndy = temp_dnd123_rot[:, 0]


    # 需要在这里把梯度重排
    dXdz1 = xy0[:, 1]
    dXdz2 = 1. / n * (dndx - dndz * dXdz1) * (1 + dXdz1 ** 2)
    dYdz1 = xy0[:, 3]
    dYdz2 = 1. / n * (dndy - dndz * dYdz1) * (1 + dYdz1 ** 2)
    # dXdz2 = 1. / n * (dndx * (1 + dXdz1 ** 2) - dndz * dXdz1)
    deriv = torch.stack([dXdz1, dXdz2, dYdz1, dYdz2], axis=1)
    n = n

    return deriv, n


def gauss_step_opl(xyz0, RefractionIndex, param):
    """
    光线追踪的高斯拟合（与本项目无关）
    Gaussian fitting for ray tracing (not relevant to this project)
    :param xyz0:
    :param RefractionIndex:
    :param param:
    :return:
    """
    Num_Ray2 = param.Num_Ray2
    step = param.step

    z0 = xyz0[:, 4]
    xy0 = xyz0[:, 0:4]

    (deriv, n) = rayeq_opl(z0, xy0, RefractionIndex, param)
    # dstepdz = torch.sqrt(1. + xy0[:, 1]**2 + xy0[:, 3]**2)
    # h = step / n / dstepdz
    h = step * torch.ones([Num_Ray2, 1])
    h = h.to(param.device)

    xyi = xy0 + deriv * h

    z4 = z0 + torch.squeeze(h)  # see above regarding squeezing
    z4 = z4.reshape([Num_Ray2, 1])
    xyzi = torch.cat([xyi, z4], axis=1)

    return xyzi, h, n


def RayTracing(xyz_init0, RefractionIndex, param):
    """
    光线追踪（与本项目无关）
    Ray tracing (not relevant to this project)
    :param xyz_init0:
    :param RefractionIndex:
    :param param:
    :return:
    """
    Num_RaySample = param.Num_RaySample
    Num_Ray2 = param.Num_Ray2

    paths = torch.zeros([Num_RaySample, Num_Ray2, 5]).to(param.device)
    xyz0 = xyz_init0
    paths[0, :, :] = xyz0

    for i in range(1, Num_RaySample):

        (xyzi, h, n) = gauss_step_opl(xyz0, RefractionIndex, param)

        xyz0 = xyzi

        paths[i, :, :] = xyzi

    return paths


def PathsMeshingInterpolation(paths, param):
    """
    射线向强度矩阵坐标的插值
    Interpolation of ray-directed intensity matrix coordinates
    :param paths: 射线路径 Ray Path
    :param param:
    :return: 射线路径引导的强度矩阵坐标插值参数 Ray path guided intensity matrix interpolated coordinate
    """
    Num_Ray = param.Num_Ray
    pathsmesh_x = paths[:, :, 0].permute(1, 0)
    pathsmesh_x = pathsmesh_x.reshape(1, 1, Num_Ray,Num_Ray,param.Num_RaySample)

    pathsmesh_y = paths[:, :, 2].permute(1, 0)
    pathsmesh_y = pathsmesh_y.reshape(1, 1, Num_Ray, Num_Ray, param.Num_RaySample)

    pathsmesh_z = paths[:, :, 4].permute(1, 0)
    pathsmesh_z = pathsmesh_z.reshape(1, 1, Num_Ray, Num_Ray, param.Num_RaySample)

    pathsmesh = torch.cat([pathsmesh_x, pathsmesh_y, pathsmesh_z], dim=1)
    pathsmesh_interp = F.interpolate(pathsmesh,
                                     size=tuple(param.PhantomSize),
                                     scale_factor=None,
                                     mode='trilinear',
                                     align_corners=True)

    return pathsmesh_interp


def Process_Parameter(param):
    """
    参数预处理 Parameter preprocessing
    :param param:
    :return:
    """
    # param = copy.copy(param)
    param.Num_Ray2 = param.Num_Ray ** 2

    param.step = 1/(param.Num_RaySample-1)

    RK4Range = param.RK4Range
    Num_RIGuassKernel = param.Num_RIGuassKernel


    temp = torch.ones([param.Num_Ray2, param.RK4Range ** 3]).to(param.device)
    param.temp_0 = temp * 0
    param.temp_1 = temp * (Num_RIGuassKernel-1)


    ## parameter process
    half = (RK4Range - 1) // 2
    Xr = np.arange(-half, half + 1)
    Yr = np.arange(-half, half + 1)
    Zr = np.arange(-half, half + 1)
    Xr, Yr, Zr = np.meshgrid(Xr, Yr, Zr)
    Xr = Xr.flatten()
    Yr = Yr.flatten()
    Zr = Zr.flatten()
    Xr = Xr.reshape([1, RK4Range ** 3])
    Yr = Yr.reshape([1, RK4Range ** 3])
    Zr = Zr.reshape([1, RK4Range ** 3])


    Xc = np.linspace(0, 1, Num_RIGuassKernel, dtype='float32')
    Yc = np.linspace(0, 1, Num_RIGuassKernel, dtype='float32')
    Zc = np.linspace(0, 1, Num_RIGuassKernel, dtype='float32')
    Xc, Yc, Zc = np.meshgrid(Xc, Yc, Zc)
    Xc = Xc.flatten()
    Yc = Yc.flatten()
    Zc = Zc.flatten()

    Xr = torch.from_numpy(Xr)
    Yr = torch.from_numpy(Yr)
    Zr = torch.from_numpy(Zr)
    Xc = torch.from_numpy(Xc)
    Yc = torch.from_numpy(Yc)
    Zc = torch.from_numpy(Zc)

    param.Xr = Xr.to(param.device)
    param.Yr = Yr.to(param.device)
    param.Zr = Zr.to(param.device)
    param.Xc = Xc.to(param.device)
    param.Yc = Yc.to(param.device)
    param.Zc = Zc.to(param.device)

    return param


def Theta2TransMatrix(param):
    """
    旋转角度转换到旋转矩阵 Convert rotation angle to rotation matrix
    :param param: 参数 parameters
    :return: 更新后的参数 Updated parameters
    """
    theta = -param.thetaY
    RotationMatrix_Y3 = np.matrix([[1, 0, 0, 0],
                                   [0, math.cos(theta), -math.sin(theta), 0],
                                   [0, math.sin(theta), math.cos(theta), 0],
                                   [0, 0, 0, 1]])
    theta = -theta
    RotationMatrix_Y3_ = np.matrix([[1, 0, 0, 0],
                                   [0, math.cos(theta), -math.sin(theta), 0],
                                   [0, math.sin(theta), math.cos(theta), 0],
                                   [0, 0, 0, 1]])

    theta = param.thetaX
    RotationMatrix_X3 = np.matrix([[math.cos(theta), 0, -math.sin(theta), 0],
                                   [0, 1, 0, 0],
                                   [math.sin(theta), 0, math.cos(theta), 0],
                                   [0, 0, 0, 1]])
    theta = -theta
    RotationMatrix_X3_ = np.matrix([[math.cos(theta), 0, -math.sin(theta), 0],
                                   [0, 1, 0, 0],
                                   [math.sin(theta), 0, math.cos(theta), 0],
                                   [0, 0, 0, 1]])

    theta = -param.thetaZ
    RotationMatrix_Z3 = np.matrix([[math.cos(theta), -math.sin(theta), 0, 0],
                                   [math.sin(theta), math.cos(theta), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
    theta = -theta
    RotationMatrix_Z3_ = np.matrix([[math.cos(theta), -math.sin(theta), 0, 0],
                                   [math.sin(theta), math.cos(theta), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    center = param.center
    ShiftMatrix_1 = np.matrix([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                              [-center[0], -center[1], -center[2], 1]])
    ShiftMatrix_2 = np.matrix([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [center[0], center[1], center[2], 1]])
    RotationMatrix_3D = RotationMatrix_Z3 * RotationMatrix_Y3 * RotationMatrix_X3
    RotationMatrix_3D_ = RotationMatrix_X3_ * RotationMatrix_Y3_ * RotationMatrix_Z3_
    param.TransMatrix_3DCrdRot = torch.from_numpy(ShiftMatrix_1 * RotationMatrix_3D * ShiftMatrix_2).float().to(param.device)
    param.TransMatrix_3DDrvRot = torch.from_numpy(RotationMatrix_3D_).float().to(param.device)

    return param


class Model_Aberration(torch.nn.Module):
    """
    像差模型 Aberration Model
    """

    def __init__(self, param):
        super().__init__()

        param = Process_Parameter(param)

        # xyz_init0 = Initialize_Ray(param.Num_Ray)
        # xyz_init0 = xyz_init0.astype('float32')
        # xyz_init0 = torch.from_numpy(xyz_init0)
        # self.xyz_init0 = xyz_init0.to(param.device)

        # self.RefractionIndex = Parameter.RefractionIndex
        # self.A = torch.nn.Parameter(RefractionIndex[:,0], requires_grad=param.FlagGrad_RefractionIndex)
        # self.Sig = RefractionIndex[:, 1]

        # self.RefractionIndex = torch.nn.Parameter(RefractionIndex, requires_grad=False)


        self.param = param

    def forward(self, Phantom, RefractionIndex, RayInit):
        """
        强度矩阵像差变换的正向过程 Forward process of intensity matrix aberration transformation
        :param Phantom: 强度矩阵 intensity matrix
        :param RefractionIndex: 折射率矩阵（本项目未使用）Refractive index matrix (not used in this project)
        :param RayInit: 射线矩阵 Ray Matrix
        :return: 包含像差的强度矩阵 Intensity matrix containing aberrations
        """
        RayInit = RayInit.reshape([self.param.Num_Ray2, 5])
        RefractionIndex = RefractionIndex.reshape([self.param.Num_RIGuassKernel**3, 2])
        self.param = Theta2TransMatrix(self.param)

        # if self.param.FlagRotate:
        #     if self.param.thetaZ != 0:
        #         Phantom = Rotation_Z(Phantom, self.param)
        #     if self.param.thetaY != 0:
        #         Phantom = Rotation_Y(Phantom, self.param)
        #     if self.param.thetaX != 0:
        #         Phantom = Rotation_X(Phantom, self.param)

        # Phantom_numpy = np.squeeze(Phantom.cpu().detach().numpy())
        # io.savemat('Phantom',
        #            {'a': 1, 'Phantom': Phantom_numpy})
        #
        # Phantom_rot_numpy = np.squeeze(Phantom_rot.cpu().detach().numpy())
        # io.savemat('Phantom_rot',
        #            {'a': 1, 'Phantom_rot': Phantom_rot_numpy})

        # Phantom = Phantom_rot.permute((2, 0, 1))



        # time_1 = time()
        paths = RayTracing(RayInit, RefractionIndex, self.param)
        # print(time() - time_1)

        # plot_Ray3D(paths.cpu().detach().numpy())

        paths = paths * 2 - 1
        # paths_numpy = paths
        # paths_numpy_x = paths_numpy[:, :, 0].reshape(25, 21, 21).permute(1, 2, 0)
        # paths_numpy_dx = paths_numpy[:, :, 1].reshape(25, 21, 21).permute(1, 2, 0)
        # paths_numpy_y = paths_numpy[:, :, 2].reshape(25, 21, 21).permute(1, 2, 0)
        # paths_numpy_dy = paths_numpy[:, :, 3].reshape(25, 21, 21).permute(1, 2, 0)
        # paths_numpy_z = paths_numpy[:, :, 4].reshape(25, 21, 21).permute(1, 2, 0)
        # TiffStackSave(paths_numpy_x, 'paths_numpy_x')
        # TiffStackSave(paths_numpy_dx, 'paths_numpy_dx')
        # TiffStackSave(paths_numpy_y, 'paths_numpy_y')
        # TiffStackSave(paths_numpy_dy, 'paths_numpy_dy')
        # TiffStackSave(paths_numpy_z, 'paths_numpy_z')


        pathsmesh_interp = PathsMeshingInterpolation(paths, self.param)
        # pathsmesh_interp_numpy = np.squeeze(pathsmesh_interp.cpu().detach().numpy())

        # pathsmesh_interp_numpy = np.squeeze(pathsmesh_interp.cpu().numpy())
        # io.savemat('pathsmesh_interp', {'pathsmesh_interp': pathsmesh_interp_numpy})

        PhantomSize = self.param.PhantomSize
        Phantom = Phantom.reshape(1, 1,
                                      PhantomSize[0],
                                      PhantomSize[1],
                                      PhantomSize[2])

        # pathsmesh_interp = pathsmesh_interp * 2 - 1
        pathsmesh_interp = pathsmesh_interp.permute(0, 2, 3, 4, 1)
        pathsmesh_interp = pathsmesh_interp[:, :, :, :, [2, 0, 1]] # z x y
        # pathsmesh_interp_0 = pathsmesh_interp[0, :, :, :, 0].squeeze()
        # pathsmesh_interp_1 = pathsmesh_interp[0, :, :, :, 1].squeeze()
        # pathsmesh_interp_2 = pathsmesh_interp[0, :, :, :, 2].squeeze()
        # TiffStackSave(pathsmesh_interp_0, 'pathsmesh_interp_0')
        # TiffStackSave(pathsmesh_interp_1, 'pathsmesh_interp_1')
        # TiffStackSave(pathsmesh_interp_2, 'pathsmesh_interp_2')

        warped_Phantom = F.grid_sample(Phantom, pathsmesh_interp,
                                       mode='bilinear',
                                       padding_mode='zeros',
                                       align_corners=True)
        # TiffStackSave(Phantom, 'Phantom')
        # TiffStackSave(warped_Phantom, 'warped_Phantom')

        return warped_Phantom, pathsmesh_interp

