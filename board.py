import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDARYINC = 5

IMAGESAVE = False

MODEL = load_model("bestmodel.h5")
PREDICT = True

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
          5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

pygame.init()

FONT = pygame.font.Font("C:/Users/Tharanga Mawan/Documents/ML Projects/Handwritten digit recognition/freesansbold.ttf", 18)

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
WHITE_INT = DISPLAYSURF.map_rgb(WHITE)
pygame.display.set_caption("Digit board")

iswriting = False

number_xcord = []
number_ycord = []

image_cnt = 1

# Initialize rect_min_x, rect_max_x, rect_min_y, and rect_max_y
rect_min_x, rect_max_x, rect_min_y, rect_max_y = 0, 0, 0, 0

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDARYINC)

            number_xcord = []
            number_ycord = []

            # Check if img_arr is non-empty and has valid dimensions
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            if img_arr.shape[0] > 0 and img_arr.shape[1] > 0:
                try:
                    # Resize img_arr to (28, 28) and reshape to (1, 28, 28, 1)
                    image = cv2.resize(img_arr, (28, 28))
                    image = image / 255.0  # Normalize to the range [0, 1]
                    image = image.reshape(1, 28, 28, 1)  # Reshape to include a single grayscale channel

                    label = str(LABELS[np.argmax(MODEL.predict(image))])
                except Exception as e:
                    print(f"Error: {e}")
                    label = "Error"
            else:
                label = "No image data"

            textSurface = FONT.render(label, True, RED, WHITE)
            textRecobj = textSurface.get_rect()
            textRecobj.left, textRecobj.bottom = rect_min_x, rect_max_y

            DISPLAYSURF.blit(textSurface, textRecobj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
