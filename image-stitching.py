import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt


# Implementation of the normalization is done with the help of
# https://stackoverflow.com/questions/52940822/what-is-the-correct-way-to-normalize-corresponding-points-before-estimation-of-f
def normalize(points):
    size = len(points)
    avg = np.average(points, axis=0)
    distances = np.subtract(points, avg)
    distances = np.power(distances, 2)
    distances = np.cumsum(distances, axis=0)
    distances = np.cumsum(distances, axis=1)
    xy_norm = math.sqrt(distances[size-1][1]) / float(size)
    diagonal_element = math.sqrt(2) / xy_norm
    element_13 = -math.sqrt(2) * avg[0] / xy_norm
    element_23 = -math.sqrt(2) * avg[1] / xy_norm
    normalization_matrix = np.array([[diagonal_element, 0, element_13], [0, diagonal_element, element_23], [0, 0, 1]])
    k_column = np.zeros((size, 1)) + 1
    new_points = np.append(points, k_column, axis=1)
    # new_points = np.zeros((size, 3)) + 1
    # new_points[:, :-1] = points
    return np.apply_along_axis(matrix_multiplication, 1, new_points, arr2=normalization_matrix)


def matrix_multiplication(arr1, arr2):
    return np.matmul(arr2, arr1)

# Implementation is done with the help of:
# https://www.reddit.com/r/computervision/comments/2h1yfj/how_to_calculate_homography_matrix_with_dlt_and/
def computeH(im1, im2):
    # Homography matrix
    n = len(im1)
    A = np.zeros((2*n, 9))
    for i in range(len(im1)):  # Using the corresponding points
        x, y, w = im1[i][0], im1[i][1], 1
        u, v, z = im2[i][0], im2[i][1], 1
        A[i * 2] = [x*z, y*z, w*z, 0, 0, 0, -x*u, -y*u, -w*u]
        A[i *2 +1] = [0, 0, 0, x*z, y*z, w*z, -x*v, -y*v, -w*v]
    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1, :] / Vh[-1,-1]
    H = np.reshape(H, (3, 3))
    return H


# Implementation is done with the help of: https://github.com/jmlipman/LAID/blob/master/IP/homography.py
def warp(im, H):
    # This part will will calculate the X and Y offsets
    bunchX = []
    bunchY = []
    height = im.shape[0]
    width = im.shape[1]

    tt = np.array([[1], [1], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0] / tmp[2])
    bunchY.append(tmp[1] / tmp[2])

    tt = np.array([[width], [1], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0]/ tmp[2])
    bunchY.append(tmp[1]/ tmp[2])

    tt = np.array([[1], [height], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0]/tmp[2])
    bunchY.append(tmp[1]/tmp[2])

    tt = np.array([[width], [height], [1]])
    tmp = np.dot(H, tt)
    bunchX.append(tmp[0]/tmp[2])
    bunchY.append(tmp[1]/tmp[2])

    refX1 = int(np.min(bunchX))
    refY1 = int(np.min(bunchY))
    refX2 = int(np.max(bunchX))
    refY2 = int(np.max(bunchY))
    ref_arr = [refX1, refX2, refY1, refY2]

    print(str(refX1) + " and " + str(refX2) + " and " + str(refY1) + " and " +str(refY2))
    # Final image whose size is defined by the offsets previously calculated
    final = np.zeros((int(refY2 - refY1), int(refX2 - refX1), 3))
    Hi = np.linalg.inv(H)
    # Iterate over the imagine to forward-transform every pixel
    for i in range(width):
        for j in range(height):
            tt = np.array([i, j, 1])
            tmp = np.dot(H, tt)
            x1 = int(tmp[0]/tmp[2] - refX1)
            y1 = int(tmp[1]/tmp[2] - refY1)
            if 0 < y1 < refY2 - refY1 and 0 < x1 < refX2 - refX1:
                final[y1][x1] = im[j][i]
    tmp_final = Image.fromarray(final.astype('uint8'), "RGB")
    tmp_final.save("_tmp_final.png")
    # Simple Interpolation
    # Interpolate empty pixels from the original image, ignoring pixels outside (extrapolating)
    print("entering for loop with shape "+ str(final.shape))
    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            if sum(final[i, j, :]) == 0:
                tt = np.array([[j + refX1], [i + refY1], [1]])
                tmp = np.dot(Hi, tt)
                x1 = round(float(tmp[0]/tmp[2]))
                y1 = round(float(tmp[1]/tmp[2]))

                if width > x1 > 0 and height > y1 > 0:
                    final[i, j, :] = im[y1, x1, :]
    print("exit for loop")
    fin_im = Image.fromarray(final.astype('uint8'), "RGB")
    fin_im.save("final1.png")
    return (fin_im, final, ref_arr)


def get_gradation_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradation_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradation_2d(start, stop, width, height, is_horizontal)

    return result


def createMask(width, height):
    array = get_gradation_3d(width, height, (130, 130, 130), (255, 255, 255), (True, True, True))
    return Image.fromarray(np.uint8(array)).save("gradation_h.jpg")

def createMaskInv(width, height):
    array = get_gradation_3d(width, height, (255, 255, 255), (130, 130, 130), (True, True, True))
    return Image.fromarray(np.uint8(array)).save("gradation_h.jpg")


def stitch_left(left2_name, left_1name, k):
    im_left1 = Image.open(left_1name)
    plt.imshow(im_left1)
    commonPoints1 = np.array(plt.ginput(k, show_clicks=True))
    print(commonPoints1)
    # commonPoints1 = [[288.18871814, 373.86042348],
    #  [669.08948652, 368.2862659],
    #  [670.94753905, 516.93046819],
    #  [544.5999671,  619.12335727],
    #  [284.47261308, 747.32898175]]

    im_left2 = Image.open(left2_name)
    plt.imshow(im_left2)
    commonPoints2 = np.array(plt.ginput(k, show_clicks=True))
    print(commonPoints2)
    # commonPoints2 = [[524.16138928, 381.2926336],
    #  [903.20410513, 370.14431842],
    #  [897.62994755, 529.93683589],
    #  [776.85653318, 622.83946233],
    #  [527.87749434, 736.18066658]]

    image_array_1 = np.array(im_left1)
    image_array_2 = np.array(im_left2)

    # # np2, np1 = normalize(commonPoints2),normalize(commonPoints1)
    # # H = computeH(np2, np1)
    H = computeH(commonPoints2, commonPoints1)
    print(commonPoints1[0])
    print(np.dot(H, [commonPoints2[0][0], commonPoints2[0][1], 1]))
    # print(np1[0])
    # print(np.dot(H, np2[0]))
    (fin_im, final, ref_arr_m) = warp(image_array_2, H)
    print(final.shape)
    print(image_array_1.shape)
    np.save('five_corresponding_points_5_wrong_left1.npy', commonPoints1)
    np.save('five_corresponding_points_5_wrong_left2.npy', commonPoints2)

    max_w = max(final.shape[1], image_array_1.shape[1]) + (image_array_1.shape[1]-ref_arr_m[1])
    max_h = max(final.shape[0], image_array_1.shape[0])
    newImage1 = Image.new('RGBA', size=(max_w, max_h), color=(0, 0, 0, 255))
    createMaskInv(final.shape[1], final.shape[0])
    mask1 = Image.open("gradation_h.jpg").convert("L")
    newImage1.paste(fin_im, mask=mask1)
    createMask(image_array_1.shape[1], image_array_1.shape[0])
    mask1 = Image.open("gradation_h.jpg").convert("L")
    newImage1.paste(im_left1, (abs(ref_arr_m[0]), abs(ref_arr_m[2])),mask=mask1)
    newImage1.save('result1.png')

def stitch_right(left1_name, left2_name, k):
    im_left1 = Image.open(left1_name)
    image_array_1 = np.array(im_left1)
    plt.imshow(im_left1)
    commonPoints1 = np.array(plt.ginput(k, show_clicks=True))
    print(commonPoints1)
    im_left2 = Image.open(left2_name)
    image_array_2 = np.array(im_left2)
    plt.imshow(im_left2)
    commonPoints2 = np.array(plt.ginput(k, show_clicks=True))
    print(commonPoints2)
    # np2, np1 = normalize(commonPoints2),normalize(commonPoints1)
    # H = computeH(np2, np1)
    H = computeH(commonPoints1, commonPoints2)
    print(commonPoints1[0])
    print(np.dot(H, [commonPoints2[0][0], commonPoints2[0][1], 1]))
    # print(np1[0])
    # print(np.dot(H, np2[0]))
    (fin_im, final, ref_arr_m) = warp(image_array_1, H)
    print(final.shape)
    print(image_array_1.shape)
    np.save('five_corresponding_points_5wrong_middle.npy', commonPoints1)
    np.save('five_corresponding_points_5wrong_left1.npy', commonPoints2)

    max_w = max(final.shape[1], image_array_2.shape[1]) + abs(image_array_2.shape[1] - ref_arr_m[1])
    max_h = max(final.shape[0], image_array_2.shape[0])
    newImage1 = Image.new('RGBA', size=(max_w, max_h), color=(0, 0, 0, 255))
    createMask(final.shape[1], final.shape[0])
    mask1 = Image.open("gradation_h.jpg").convert("L")
    newImage1.paste(fin_im,(abs(max_w- final.shape[1]),0), mask=mask1)
    createMaskInv(image_array_2.shape[1], image_array_2.shape[0])
    mask1 = Image.open("gradation_h.jpg").convert("L")
    newImage1.paste(im_left2, (max_w - final.shape[1] - abs(ref_arr_m[0]), abs(ref_arr_m[2])),mask=mask1)
    newImage1.save('result2.png')

stitch_left("left-2.jpg", "left-1.jpg", 5)
stitch_right("middle.jpg", "left-1.jpg", 5)