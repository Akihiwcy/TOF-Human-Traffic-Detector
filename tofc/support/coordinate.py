def converter(bounds, kernel_size, stride, padding=0):
    """
    Calculate the ordinary coordinates of the point
    :param bounds: the coordinate (x, y, w, h, area) after pooling
    :param stride: the stride of pooling kernel,
    :param kernel_size: the kernel size
    :param padding: the padding strategy
    :return: Original coordinates (x_original, y_original)
    """
    # xx012 12345  45678  padding=2, kernel_size=5, stride=3
    #   0     1      2
    # Get the original coordinate which is in the first two columns
    #     print('bounds: ', bounds)
    bounds[:, 0:2] = bounds[:, 0:2] * stride + (kernel_size // 2) - padding
    #     print('bounds_convert: ', bounds)
    #     print()

    return bounds
