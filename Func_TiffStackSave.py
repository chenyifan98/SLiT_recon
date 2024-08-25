
import numpy as np

from libtiff import TIFF


def TiffStackSave(Phantom, addr):
    """
    保存强度矩阵 Save the intensity matrix
    :param Phantom: 强度矩阵 intensity matrix
    :param addr: 路径 path
    :return:
    """
    Phantom = np.squeeze(Phantom.cpu().detach().numpy())

    tif = TIFF.open(addr + '.tif', mode='w')
    for i in range(0, Phantom.shape[2]):
        tif.write_image(np.squeeze(Phantom[:, :, i]))
    tif.close()

    return

def TiffStackLoad(PhantomSize, addr):
    """
    读取强度矩阵  load the intensity matrix
    :param PhantomSize: 强度矩阵尺寸 intensity matrix size
    :param addr: 路径 path
    :return: 强度矩阵
    """
    # addr = './RefractionTestData/Phantom_AVG_2'
    tif_Wigner = TIFF.open(addr + '.tif', mode='r')

    Image_stack = np.zeros(PhantomSize)
    i = 0
    for image in tif_Wigner.iter_images():
        Image_stack[:, :, i] = image
        i = i + 1

    return Image_stack

