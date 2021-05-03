import numpy as np
import cv2
import random
from support import show_seg


def water_filling_simple(depth_img, K):
    """
    This Function is used to simulate the rain drops to find out the local minimum, which may be a head.
    :param depth_img: Input depth image with the size of M x N
    :param K: The number of rain drops
    :return: Measure Function g
    """
    try:
        M = len(depth_img)
        N = len(depth_img[0])
    except IndexError:
        print('Error in water_filling_simple: Index out of image range')
        return

    g = np.zeros((M, N))  # g(x,y) is the measure function/img
    # img_reverse = g - depth_img
    # print(img_reverse)

    for k in range(K):

        # print('k: ', k)

        x = random.randint(0, M - 1)
        y = random.randint(0, N - 1)
        # count = 0
        while True:
            # 在不越界的前体下，找到4-邻域中最深的地方
            min_depth = d = 0
            min_loc = [0, 0]

            if depth_img[x, y] == 255:  # 避免无用的水滴
                break

            if x - 1 >= 0:
                d = depth_img[x - 1, y] + g[x - 1, y] - (depth_img[x, y] + g[x, y])
                if d < min_depth:
                    min_depth = d
                    min_loc[0], min_loc[1] = x - 1, y
            if x + 1 <= M - 1:
                d = depth_img[x + 1][y] + g[x + 1][y] - (depth_img[x][y] + g[x][y])
                if d < min_depth:
                    min_depth = d
                    min_loc[0], min_loc[1] = x + 1, y
            if y - 1 >= 0:
                d = depth_img[x][y - 1] + g[x][y - 1] - (depth_img[x][y] + g[x][y])
                if d < min_depth:
                    min_depth = d
                    min_loc[0], min_loc[1] = x, y - 1
            if y + 1 <= N - 1:
                d = depth_img[x][y + 1] + g[x][y + 1] - (depth_img[x][y] + g[x][y])
                if d < min_depth:
                    min_depth = d
                    min_loc[0], min_loc[1] = x, y - 1

            # 更新水滴位置
            if min_depth < 0:
                x, y = min_loc[0], min_loc[1]
                # count += 1
                # print('count:', count)
            else:
                g[x][y] += 1
                break

    return g


def water_filling_fast(f, K, R, r):
    """
    Fast water filling
    :param f: Depth image
    :param K: Number of raindrops
    :param R: Amount of water in one raindrop
    :param r: Amount of water dropped one time
    :return: Mesure function/img g
    """
    try:
        M = len(f)
        N = len(f[0])
    except IndexError:
        print('Error in water_filling_fast: Index out of image range')
        return

    g = np.zeros((M, N))  # g(x,y) is the measure function/img
    #     img_reverse = g - f

    for k in range(K):
        x = random.randint(0, M - 1)
        y = random.randint(0, N - 1)

        if f[x, y] == 255:  # 避免无用的水滴
            continue
        #         print('k: ', k)
        w = R
        while w > 0:
            # 在不越界的前体下，找到4-邻域中最深的地方
            min_depth = d = 0
            min_loc = [0, 0]
            if x - 1 >= 0:
                d = f[x - 1][y] + g[x - 1][y] - (f[x][y] + g[x][y])
                if d < min_depth:
                    min_depth = d
                    min_loc[0], min_loc[1] = x - 1, y
            if x + 1 <= M - 1:
                d = f[x + 1][y] + g[x + 1][y] - (f[x][y] + g[x][y])
                if d < min_depth:
                    min_depth = d
                    min_loc[0], min_loc[1] = x + 1, y
            if y - 1 >= 0:
                d = f[x][y - 1] + g[x][y - 1] - (f[x][y] + g[x][y])
                if d < min_depth:
                    min_depth = d
                    min_loc[0], min_loc[1] = x, y - 1
            if y + 1 <= N - 1:
                d = f[x][y + 1] + g[x][y + 1] - (f[x][y] + g[x][y])
                if d < min_depth:
                    min_depth = d
                    min_loc[0], min_loc[1] = x, y - 1
            if min_depth + r < 0:
                x, y = min_loc[0], min_loc[1]
            else:
                g[x][y] = g[x][y] + min(r, w)
                w -= r
    return g


def human_bounding(g, threshold, connectivity=4, debugMode=1):
    '''
    This function using measure img/function g(x,y) and threshold to find the individual.
    :param g: Measure function/img g(x,y) from water filling.
    :param threshold: Threshold to remove small object
    :param connectivity: 4 or 8 connected (default: 4)
    :param debugMode: Using Mode
    :return: Image with size MxN that represents the bonding of individual.
    '''
    if debugMode:
        cv2.namedWindow('g', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('g', 320, 240)
        cv2.imshow('g', g)

    g[g < threshold] = 0
    g_thresholded = g
    if debugMode:
        cv2.namedWindow('g_thresholded', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('g_thresholded', 320, 240)
        cv2.imshow('g_thresholded', g_thresholded)

    count = 0
    boundings = []
    centers = []
    if np.max(g_thresholded) > 0:
        g_thresholded[g_thresholded > 0] = 1
        g_thresholded = g_thresholded.astype('uint8')
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(g_thresholded, connectivity)

        labels = labels.astype('uint8')
        img_label = show_seg.show_segmentation(labels, num_labels)

        #     去掉过小的区域
        for i in range(num_labels):
            x, y, w, h, area = stats[i]
            if area < 3 or area > 50:
                continue
            else:
                # 将符合条件的区域加入到检测结果中
                count += 1
                boundings.append(stats[i])
                centers.append(centroids[i])

                if debugMode:
                    cv2.rectangle(img_label, (x, y), (x + w, y + h), 255)
                    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Detection', 320, 240)
                    cv2.imshow('Detection', img_label)
    #     print('count:{}, centers:{}'.format(count, len(centers)))
    return count, boundings, centers


def human_detection(img, K, threshold, connectivity=4, debugMode=1):
    g = water_filling_simple(img, K)
    count, bounds, centers = human_bounding(g, threshold, connectivity, debugMode)
    return count, bounds, centers
