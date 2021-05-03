import numpy as np
import cv2


def get_foreground(frame, bg, t_low=110, t_high=200):
    """
    Used to get the foreground image of the frame
    :param frame: The frame needed to be handle
    :param bg: Background
    :param t_low: Low threshold to eliminate the noise
    :param t_high: High threshold to eliminate the noise
    :return: Foreground image
    """
    assert type(frame) is np.ndarray
    assert type(bg) is np.ndarray
    foreground_mask = frame - bg
    foreground_mask[foreground_mask < t_low] = 0
    foreground_mask[foreground_mask >= t_high] = 0
    foreground_mask[foreground_mask != 0] = 1

    img_foreground = cv2.medianBlur(frame * foreground_mask, 9)
    img_foreground[img_foreground == 0] = 255

    return img_foreground
