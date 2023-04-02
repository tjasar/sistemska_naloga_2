from math import sqrt

import cv2
import numpy as np


# Nad originalno sliko uporabimo vertikalni in horizontalni kernel velikosti 3x3
def apply_filter_3x3(img, x_kernel, y_kernel):
    # naredimo izhodno sliko velikosti vhodne slike in jo napolnimo z 0
    img_output = np.zeros(img.shape, np.uint8)
    # naredimo sliko za x filter velikosti vhodne slike in napolnimo z 0
    x_filter = np.zeros(img.shape, np.uint8)
    # naredimo sliko za y filter velikosti vhodne slike in napolnimo z 0
    y_filter = np.zeros(img.shape, np.uint8)

    # se premikamo cez vhodno sliko piksel po piksel in racunamo s pomocjo kernelov novo vrednost pikla na isti lokaciji
    # na izhodni sliki, robe zanemarimo da nimamo problemov z nastavljanjem kernelov
    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            # poracunamo vrednost pixla s pomocjo kernela in vzamemo absolutno vrednost, tako naredimo za x in y
            # filter
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

            # vrednosti x in y filtra za pixel nato zdruzimo in shranimo v izhodno sliko
            img_output[y][x] = sqrt(pow(y_filter[y][x], 2) + pow(x_filter[y][x], 2))

    return img_output


# Enako kot zgornja funkcija le da deluje z 2x2 kernelom
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


# dolocenje x in y kernela za roberts cross in konvolucija 2x2
def my_roberts(img):
    roberts_x_kernel = np.array([[1, 0],
                                 [0, -1]])

    roberts_y_kernel = np.array([[0, 1],
                                 [-1, 0]])

    return apply_filter_2x2(img, roberts_x_kernel, roberts_y_kernel)


# dolocenje x in y kernela za prewitt in konvolucija 3x3
def my_prewitt(img):
    prewitt_x_kernel = np.array([[1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1]])

    prewitt_y_kernel = np.array([[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]])

    return apply_filter_3x3(img, prewitt_x_kernel, prewitt_y_kernel)


# dolocenje x in y kernela za sobel in konvolucija 3x3
def my_sobel(img):
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_y_kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])

    return apply_filter_3x3(img, sobel_x_kernel, sobel_y_kernel)


# canny filter iz knjiznice cv2
def canny(img, sp_prag, zg_prag):
    return cv2.Canny(img, sp_prag, zg_prag)


# spreminjanje kontrasta slike glede na podano alfa in beta vrednost
def spremeni_kontrast(img, alfa, beta):
    img = np.clip((alfa * img.astype(float) + beta), 0, 255).astype(np.uint8)
    return img


# funkcija za prikazovanje fotografije
def showImage(name, img):
    cv2.imshow(name, img)


def main():
    img = cv2.imread("lenna.png", 0)
    original = img

    # prikaz razlicnih slik z razlicno uporabljenimi detektorji robov
    showImage("Slika", original)
    showImage("Slika - glajena", cv2.GaussianBlur(original, (5, 5), 0))
    showImage("Slika - kontrast svetlo", spremeni_kontrast(img, 2.8, -30))
    showImage("Slika - kontrast temno", spremeni_kontrast(img, 1.2, -70))

    showImage("Slika Roberts - original", my_roberts(img))
    showImage("Slika Roberts - kontrast svetlo", my_roberts(spremeni_kontrast(img, 2.8, -30)))
    showImage("Slika Roberts - kontrast temno", my_roberts(spremeni_kontrast(img, 1.2, -70)))

    showImage("Slika Sobel - original", my_sobel(img))
    showImage("Slika Sobel - kontrast svetlo", my_sobel(spremeni_kontrast(img, 2.8, -30)))
    showImage("Slika Sobel - kontrast temno", my_sobel(spremeni_kontrast(img, 1.2, -70)))

    showImage("Slika Prewitt - original", my_prewitt(img))
    showImage("Slika Prewitt - kontrast svetlo", my_prewitt(spremeni_kontrast(img, 2.8, -30)))
    showImage("Slika Prewitt - kontrast temno", my_prewitt(spremeni_kontrast(img, 1.2, -70)))

    showImage("Slika Canny - original", canny(img, 50, 150))
    showImage("Slika Canny - kontrast svetlo", canny(spremeni_kontrast(img, 2.8, -30), 50, 150))
    showImage("Slika Canny - kontrast temno", canny(spremeni_kontrast(img, 1.2, -70), 50, 150))

    showImage("Slika Sobel - original", my_sobel(img))
    showImage("Slika Sobel - glajena", my_sobel(cv2.GaussianBlur(img, (5, 5), 0)))

    showImage("Slika Canny - original", canny(img, 50, 150))
    showImage("Slika Canny - glajena", canny(spremeni_kontrast(cv2.GaussianBlur(img, (5, 5), 0), 1.2, -70), 50, 150))

    cv2.waitKey()


if __name__ == "__main__":
    main()
