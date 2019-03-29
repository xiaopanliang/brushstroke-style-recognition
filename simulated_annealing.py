from random import randint, uniform
import cv2
import math
import os
import numpy as np


dirName = "tmp_tests/"
k_max = 100000000000


def temperature(r):
    return 1 - r


# pt in (x, y)
def random_pick_pt(height, width):
    return randint(0, width - 1), randint(0, height - 1)


def get_neighbors(pt, mx_h, mx_w):
    x = pt[0]
    y = pt[1]

    nb1 = (x + 1, y)
    nb2 = (x - 1, y)
    nb3 = (x, y + 1)
    nb4 = (x, y - 1)
    nb5 = (x + 1, y + 1)
    nb6 = (x + 1, y - 1)
    nb7 = (x - 1, y + 1)
    nb8 = (x - 1, y - 1)

    nbs = [nb1, nb2, nb3, nb4, nb5, nb6, nb7, nb8]
    ans = []
    for nb in nbs:
        x = nb[0]
        y = nb[1]
        if (0 <= x < mx_w) and (0 <= y < mx_h):
            ans.append(nb)

    return ans


def difference(pt, ns, img):
    p_px = img[pt[1]][pt[0]]
    total = 0
    for n in ns:
        n_px = img[n[1]][n[0]]
        diff_r = math.sqrt(
            math.pow(p_px[0] - n_px[0], 2) +
            math.pow(p_px[1] - n_px[1], 2) +
            math.pow(p_px[2] - n_px[2], 2)) / math.sqrt(3 * math.pow(0xff, 2))
        total += diff_r
    return total / len(ns)


def swap(p1, p2, img):
    tmp_px = img[p1[1]][p1[0]]
    img[p1[1]][p1[0]] = img[p2[1]][p2[0]]
    img[p2[1]][p2[0]] = tmp_px


def P(orig, new):
    return orig - new


def simulated_annealing(img):
    height, width, _ = img.shape
    img.flags.writeable = True
    for k in range(0, k_max):

        p1 = random_pick_pt(height, width)
        p2 = random_pick_pt(height, width)

        n1 = get_neighbors(p1, height, width)
        n2 = get_neighbors(p2, height, width)

        orig = difference(p1, n1, img) + difference(p2, n2, img)
        new = difference(p1, n2, img) + difference(p2, n1, img)

        if P(orig, new) > 0:
            swap(p1, p2, img)
        if k % 5000 == 0:
            print(k)
        if k % 1000000 == 0:
            cv2.imwrite(dirName + "test_result_" + str(k) + ".jpg", img)

    return img


if not os.path.exists(dirName):
    os.makedirs(dirName)
img = cv2.imread("test.jpg")

# Reshape the image and shuffle pixels
height, width, depth = img.shape
img = np.reshape(img, [height * width, depth])
np.random.shuffle(img)

# Reshape the image back
img = np.reshape(img, [height, width, depth])

img = simulated_annealing(img)
cv2.imwrite("test_result.jpg", img)
