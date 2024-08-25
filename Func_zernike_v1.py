import numpy as np
from math import factorial as fact


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def zernike_pol(rho, theta, n, m):
    """Evaluate the normalized radial zernike polynomials in the unit
    circle using the radial and azimuthal index convention outlined in
    https://en.wikipedia.org/wiki/Zernike_polynomials.

    Arguments:
    rho -- radial mesh covering the unit range [0, 1]
    theta -- polar angle mesh covering the range [0, 2*pi]
    n, m -- integer radial Zernike polynomial indices
    """

    def R(r, i, j):
        radial = 0
        for s in range((i - np.abs(j)) // 2 + 1):
            num = (-1) ** s * fact(i - s) * r ** (i - 2 * s)
            den = fact(s) * fact((i + np.abs(j) - 2 * s) // 2) * fact((i - np.abs(j) - 2 * s) // 2)
            radial += num / den
        return radial

    def Norm(i, j):
        if j == 0:
            return np.sqrt(2 * (i + 1) / (1 + 1))
        else:
            return np.sqrt(2 * (i + 1))

    amplitude = Norm(n, m) * R(rho, n, m)
    if m < 0:
        return amplitude * np.sin(m * theta)
    else:
        return amplitude * np.cos(m * theta)


def zernike_dictionary(rank, size):

    rate = 0.707
    x = np.linspace(-1, 1, size[0]) * rate
    y = np.linspace(-1, 1, size[1]) * rate
    X, Y = np.meshgrid(x, y)
    rho, theta = cart2pol(X, Y)

    itemNum = int((rank+2)*(rank+1)/2)
    ZernikeDictionary = np.zeros([size[0], size[1], itemNum])
    for n in range(0, rank+1):
        for m in range(-n, n+1, 2):
            temp = int((n+1)*n/2+(n+m)/2)
            ZernikeDictionary[:, :, temp] = zernike_pol(rho, theta, n, m)

    return ZernikeDictionary
