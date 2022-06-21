import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def lanesDetection(img):
    # img = cv.imread("./img/road.png")
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

   # print(img.shape) # returneaza dimensiunile imaginii
    height = img.shape[0]
    width = img.shape[1]

    region_of_interest_vertices = [
        (200, height), (width/2, height/1.37), (width, height)
    ]
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY) # rgb to gray
    edge = cv.Canny(gray_img, 50, 100, apertureSize=3)
    cropped_image = region_of_interest(
        edge, np.array([region_of_interest_vertices], np.int32)) # imaginea decupata

    lines = cv.HoughLinesP(cropped_image, rho=2, theta=np.pi/180,
                           threshold=50, lines=np.array([]), minLineLength=10, maxLineGap=30) # detecteaza liniile din imaginea decupata
    image_with_lines = draw_lines(img, lines)
    #plt.imshow(image_with_lines)
    #plt.show()
    return image_with_lines

# defineste regiunea de interes din imagine
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = (255)
    cv.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

# traseaza liniile pe imaginea originala
def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def videoLanes():
    cap = cv.VideoCapture('./img/v1.mp4') # este preluat videoclipul din folder
    while(cap.isOpened()):
        ret, frame = cap.read() # memoreaza frame-ul
        frame = lanesDetection(frame) # apeleaza functia avand ca parametru frame-ul respectiv
        cv.imshow('Detectia benzilor de circulatie', frame) # afiseaza frame-urileq
        if cv.waitKey(1) & 0xFF == ord('q'): # se inchide programul prin apasarea tastei Q
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    videoLanes()
