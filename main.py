from math import sqrt

import cv2
import numpy as np


# Nad originalno sliko uporabimo vertikalni in horizontalni kernel
def apply_filter_3x3(img, x_kernel, y_kernel):
    img_output = np.zeros(img.shape, np.uint8)
    x_filter = np.zeros(img.shape, np.uint8)
    y_filter = np.zeros(img.shape, np.uint8)

    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            x_filter[y][x] = np.absolute((
                    x_kernel[0][2] * img[y - 1][x + 1] +
                    x_kernel[1][2] * img[y][x + 1] +
                    x_kernel[2][2] * img[y + 1][x + 1] +
                    x_kernel[0][0] * img[y - 1][x - 1] +
                    x_kernel[1][0] * img[y][x - 1] +
                    x_kernel[2][0] * img[y + 1][x - 1]
            ))
            y_filter[y][x] = np.absolute((
                    y_kernel[0][0] * img[y - 1][x - 1] +
                    y_kernel[0][1] * img[y - 1][x] +
                    y_kernel[0][2] * img[y - 1][x + 1] +
                    y_kernel[2][0] * img[y + 1][x - 1] +
                    y_kernel[2][1] * img[y + 1][x] +
                    y_kernel[2][2] * img[y + 1][x + 1]
            ))

            img_output[y][x] = sqrt(pow(y_filter[y][x], 2) + pow(x_filter[y][x], 2))

    return img_output

# Enako kot zgornja funkcija le da deluje nad 2x2 kernelom
def apply_filter_2x2(img, x_kernel, y_kernel):
    img_output = np.zeros(img.shape, np.uint8)
    x_filter = np.zeros(img.shape, np.uint8)
    y_filter = np.zeros(img.shape, np.uint8)

    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            x_filter[y][x] = np.absolute((
                    x_kernel[0][1] * img[y - 1][x] +
                    x_kernel[1][1] * img[y][x] +
                    x_kernel[1][0] * img[y][x - 1] +
                    x_kernel[0][0] * img[y - 1][x - 1]
            ))
            y_filter[y][x] = np.absolute((
                    y_kernel[0][0] * img[y - 1][x - 1] +
                    y_kernel[0][1] * img[y - 1][x] +
                    y_kernel[1][1] * img[y][x] +
                    y_kernel[1][0] * img[y][x - 1]
            ))

            img_output[y][x] = sqrt(pow(y_filter[y][x], 2) + pow(x_filter[y][x], 2))

    return img_output


def my_roberts(img):
    roberts_x_kernel = np.array([[1, 0],
                                 [0, -1]])

    roberts_y_kernel = np.array([[0, 1],
                                 [-1, 0]])

    return apply_filter_2x2(img, roberts_x_kernel, roberts_y_kernel)


def my_prewitt(img):
    prewitt_x_kernel = np.array([[1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1]])

    prewitt_y_kernel = np.array([[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]])

    return apply_filter_3x3(img, prewitt_x_kernel, prewitt_y_kernel)


def my_sobel(img):
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_y_kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])

    return apply_filter_3x3(img, sobel_x_kernel, sobel_y_kernel)


def canny(slika, sp_prag, zg_prag):
    # vaša implementacija
    return None


def spremeni_kontrast(slika, alfa, beta):
    pass


def showImage(name, img):
    cv2.imshow(name, img)


def main():
    img = cv2.imread("lenna.png", 0)
    original = img

    showImage("Slika", original)

    showImage("Slika Roberts", my_roberts(img))

    showImage("Slika Sobel", my_sobel(img))

    showImage("Slika Prewitt", my_prewitt(img))

    cv2.waitKey()


if __name__ == "__main__":
    main()
