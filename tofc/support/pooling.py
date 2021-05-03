import numpy as np
from numpy.lib.stride_tricks import as_strided


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    """
    2D Pooling, n = (len + 2*padding - size) / stride + 1
    :param A: input 2D array
    :param kernel_size: int, the size of the window
    :param stride: int, the stride of the window
    :param padding: int, implicit zero paddings on both sides of the input
    :param pool_mode: string, 'max' or 'avg'
    :return: image after pooling
    """

    # Padding
    A = np.pad(A, padding, mode='constant')
    # print('A: ', A)

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride * A.strides[0],
                              stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)
