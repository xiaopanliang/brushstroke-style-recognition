import numpy as np
import cv2


def update_col(col_forward, tmp_col):
    if col_forward:
        tmp_col += 1
    else:
        tmp_col -= 1
    return tmp_col


def sort_pixel_r(empty_img, sorted_pixels, pixel_index, height, width, row, col, row_forward, col_forward):
    tmp_row = row
    tmp_col = col
    cur_index = 0
    max_index = pixel_index + 3 * (width - 1)
    if max_index > len(sorted_pixels):
        max_index = len(sorted_pixels) - 1
    # Copy pixels to the current frame, each frame has the height 3
    for i in range(pixel_index, max_index):
        empty_img[tmp_row][tmp_col] = sorted_pixels[i]
        # Determine if the direction is forward or backward for the row
        if row_forward and tmp_row < row + 2:
            # If the row is less than the upper frame 3
            # Increase the row normally
            if tmp_row < height - 1:
                tmp_row += 1
            else:
                # If the row reaches the max height, stops updating the variable
                # Instead, we should move to next col
                tmp_col = update_col(col_forward, tmp_col)
                row_forward = not row_forward
        elif not row_forward and tmp_row > row:
            # Decrease the row normally
            tmp_row -= 1
        elif tmp_row == row:
            # If the row reaches the upper limit of the frame
            # Bounce it back
            tmp_col = update_col(col_forward, tmp_col)
            row_forward = True
        elif tmp_row == row + 2:
            # If the row reaches the lower limit of the frame
            # Bounce it back
            tmp_col = update_col(col_forward, tmp_col)
            row_forward = False
        cur_index = i
    # Copy last row elements
    tmp_row = row
    cur_index += 1
    max_index = cur_index + 3
    # Last the col of the current frame should start from the original
    # row and keep copying until the start of the next 3 rows frame
    if max_index > len(sorted_pixels):
        max_index = len(sorted_pixels)
    for i in range(cur_index, max_index):
        empty_img[tmp_row][tmp_col] = sorted_pixels[i]
        tmp_row += 1
        cur_index = i
    cur_index += 1
    if cur_index < len(sorted_pixels):
        sort_pixel_r(empty_img, sorted_pixels, cur_index, height, width, tmp_row, tmp_col, True, not col_forward)


def sort_pixels(img):
    empty_mat = np.zeros(img.shape, dtype=img.dtype)
    height, width = img.shape
    img = np.reshape(img, [-1, ])
    sorted_pxs = np.sort(img)
    sort_pixel_r(empty_mat, sorted_pxs, 0, height, width, 0, 0, True, True)
    return empty_mat


if __name__ == '__main__':
    img = cv2.imread("henri-edmond-cross_a-garden-in-provence-1901.jpg")
    img = cv2.resize(img, (512, 512))  # The image is read as BGR format
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]
    # cv2.imwrite("copy.jpg", red_channel)
    # a = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
    #               [9, 10, 11, 12, 13, 14, 15, 16],
    #               [17, 18, 19, 20, 21, 22, 23, 24],
    #               [25, 26, 27, 28, 29, 30, 31, 32],
    #               [33, 34, 35, 36, 37, 38, 39, 40],
    #               [41, 42, 43, 44, 45, 46, 47, 48],
    #               [49, 50, 51, 52, 53, 54, 55, 56]])
    blue_channel_result = sort_pixels(blue_channel)
    green_channel_result = sort_pixels(green_channel)
    red_channel_result = sort_pixels(red_channel)
    # cv2.imwrite("copy.jpg", result)
    # print(result)
