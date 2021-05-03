import numpy as np


def show_segmentation(labelImg, labelNum=0):
    """
    It is used to show the RGB version of the label image
    :param labelImg: Label image
    :param labelNum: The max number of label types
    :return: RGB label image
    """
    if not labelNum:
        labelNum = max(labelImg)
    m, n = len(labelImg), len(labelImg[0])
    output = np.zeros((m, n, 3), np.uint8)
    for i in range(0, labelNum + 1):
        mask = labelImg == i
        output[:, :, 0][mask] = np.random.randint(0, 255)
        output[:, :, 1][mask] = np.random.randint(0, 255)
        output[:, :, 2][mask] = np.random.randint(0, 255)
        # cv2.imshow('Segmentation', output)
    return output
