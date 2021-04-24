import cv2
import time
import collections
import numpy as np

from handler import img_recover, background

PATH = '../../Data/'
FILE = '1556158515.bin'
FRAME_RATE = 10 # 0.01S
IMAGE_LENGTH = 320
IMAGE_WIDTH = 240
# 1455208110一人一盒  1556075168一人  1556158515 20181214153330_14_13_uint16_amp0
# 20181226180237_15_15_uint16_amp0(存在问题) 20190124114254_12_14_uint16_amp0


def display(vid='others', verbose=False):
    print('{} video is displaying now!'.format(vid.title()))

    if vid.lower() == 'others':
        file_path = PATH + vid.lower() + '/' + FILE
        file = open(file_path, 'rb')
    elif vid.lower() == 'own':
        capture = cv2.VideoCapture(PATH+FILE)

    bg = background.Background(length=20)

    cnt = 0
    try:
        while 1:
            cnt += 1
            if vid.lower() == 'others':
                frame = np.frombuffer(file.read(IMAGE_LENGTH * IMAGE_WIDTH * 2), dtype=np.uint16)
                img = frame.copy().reshape(IMAGE_WIDTH, IMAGE_LENGTH)
                img = (img.copy() / 20).astype(np.uint8)
            elif vid.lower() == 'own':
                _, frame = capture.read()
                img = img = frame[:, :, 0]
                """
                剩下的一些处理，降采样等，未完待续
                """

            # Before handle the video
            # cv2.imshow("img", img)
            img_rgb = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            # cv2.imshow("img_rgb", img_rgb)

            img_re = img_recover.basic_recover(img)
            cv2.imshow("vid", np.hstack([img, img_re]))

            img_rgb_re = cv2.applyColorMap(img_re, cv2.COLORMAP_JET)
            cv2.imshow("vid_rgb", np.hstack([img_rgb, img_rgb_re]))

            if bg.update(img_re):
                cv2.imshow("background", bg.bg)

            cv2.waitKey(FRAME_RATE)

    except ValueError:
        cv2.destroyAllWindows()

