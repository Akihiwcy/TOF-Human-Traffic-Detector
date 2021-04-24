import cv2


def basic_recover(frame, adv=False):
    """
    Used to recover the frame from other dataset
    :param frame: input frame
    :param adv: advanced idea, default=False
    :return: frame after recover
    """
    # frame = cv2.bilateralFilter(frame, 6, 20, 20)
    return cv2.medianBlur(frame, 9)
