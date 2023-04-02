import cv2


def showImage(name, img):
    cv2.imshow(name, img)


def main():
    img = cv2.imread("primer_iz_vaj/lenna.png", 0)
    original = img

    showImage("SLika", img)
    cv2.waitKey()


if __name__ == "__main__":
    main()
