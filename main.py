import cv2


def my_roberts(slika):
    # vaša implementacija
    return None


def my_prewitt(slika):
    # vaša implementacija
    return None


def my_sobel(slika):
    # vaša implementacija
    return None


def canny(slika, sp_prag, zg_prag):
    # vaša implementacija
    return None


def spremeni_kontrast(slika, alfa, beta):
    pass


def showImage(name, img):
    cv2.imshow(name, img)


def main():
    img = cv2.imread("primer_iz_vaj/lenna.png", 0)
    original = img

    showImage("SLika", img)
    cv2.waitKey()


if __name__ == "__main__":
    main()
