import cv2
import time
import collections
import numpy as np

from handler import img_recover, background, foreground
from support import pooling, coordinate
from detect import water_filling
from track import track

PATH = '../../Data/'
FILE = '1556158515.bin'
FRAME_RATE = 10 # 0.01S
IMAGE_LENGTH = 320
IMAGE_WIDTH = 240
# 1455208110一人一盒  1556075168一人  1556158515 20181214153330_14_13_uint16_amp0
# 20181226180237_15_15_uint16_amp0(存在问题) 20190124114254_12_14_uint16_amp0


def display(vid='others', verbose=True):
    print('{} video is displaying now!'.format(vid.title()))

    if vid.lower() == 'others':
        file_path = PATH + vid.lower() + '/' + FILE
        file = open(file_path, 'rb')
    elif vid.lower() == 'own':
        capture = cv2.VideoCapture(PATH+FILE)
    else:
        raise Exception("Invalid Video Name!", vid)

    # get the background
    bg = background.Background(length=20)

    cnt = 0
    try:
        tracks = []  # walking tracks
        people_count = 0  # the number of people who are in the room
        people_num = 0  # the number of people who are in the frame
        people_num_old = 0

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
                TODO: 剩下的一些处理，降采样等，未完待续
                """
            else:
                raise Exception("Invalid Video Name!", vid)

            # Show the frame before preprocess
            img_rgb = cv2.applyColorMap(img, cv2.COLORMAP_JET)

            img_re = img_recover.basic_recover(img)
            cv2.imshow("vid", np.hstack([img, img_re]))

            if verbose:
                img_rgb_re = cv2.applyColorMap(img_re, cv2.COLORMAP_JET)
                cv2.imshow("vid_rgb", np.hstack([img_rgb, img_rgb_re]))

            if bg.update(img_re):
                # get the foreground part of frame
                img_foreground = foreground.get_foreground(img_re, bg.bg)

                # use pooling to downsize the frame in order to speed up
                kernel_size = 9
                stride = 9
                padding = 0

                img_pooling = pooling.pool2d(img_foreground, kernel_size, stride, padding, pool_mode='avg')
                obj_num, obj_bounds, obj_centers = water_filling.human_detection(img_pooling,
                                                                                 K=2740,
                                                                                 threshold=6,
                                                                                 connectivity=4,
                                                                                 debugMode=0)
                obj_centers = np.array(obj_centers, dtype=np.uint16)

                if obj_num:
                    obj_centers_org = coordinate.converter(obj_centers, kernel_size, stride, padding).tolist()

                    if obj_centers_org:
                        # Try to make the track
                        tracks = track.match(old_tracks=tracks,
                                             centers=obj_centers_org,
                                             k=8,
                                             verbose=False)
                        people_num = 0
                        for i in range(len(obj_centers_org)):
                            x, y = obj_centers_org[i]
                            # The width of the bounding
                            w = 10
                            cv2.rectangle(img_foreground, (x - w, y - w), (x + w, y + w), 0)
                            if (IMAGE_WIDTH / 3) < y < (IMAGE_WIDTH * (2/3)):
                                people_num += 1

                        # Draw tracks
                        if len(tracks) > 0:
                            for t in tracks:
                                old_index, new_index = 0, 1
                                if len(t) > 3 and t[-1][1] >= IMAGE_WIDTH / 2 >= t[-2][1]:
                                    people_count += 1
                                    # print('TrackToCount:', tracks[i])
                                    # print('count + : ', people_count)
                                    # print('------------------------------------')
                                elif len(t) > 3 and t[-1][1] <= IMAGE_WIDTH / 2 <= t[-2][1]:
                                    people_count -= 1
                                    # print('TrackToCount:', tracks[i])
                                    # print('count - : ', people_count)
                                    # print('====================================')
                                people_count = max(0, people_count)
                                while len(t) > 1 and new_index < len(t):
                                    cv2.line(img_foreground,
                                             tuple(t[old_index]),
                                             tuple(t[new_index]), 0)
                                    old_index, new_index = old_index + 1, new_index + 1

                # 绘制统计基准线
                cv2.line(img_foreground, (0, 80), (320, 80), 200)
                cv2.line(img_foreground, (0, 160), (320, 160), 200)
                cv2.line(img_foreground, (0, 120), (320, 120), 0)

                cv2.putText(img_foreground,
                            "In the Room: {}".format(people_count),
                            (25, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
                cv2.putText(img_foreground,
                            "In the Counting Area: {}".format(people_num),
                            (25, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)

                if verbose:
                    cv2.imshow("get foreground", np.hstack([bg.bg, img_re-bg.bg, img_foreground]))
                else:
                    cv2.imshow('img_foreground', img_foreground)

            cv2.waitKey(FRAME_RATE)

    except ValueError:
        cv2.destroyAllWindows()

