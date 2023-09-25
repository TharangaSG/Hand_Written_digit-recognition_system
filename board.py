import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWNSIZEX = 640
WINDOWNSIZEY = 480
BOUNDARYINC = 5

image_cnt = 1

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PREDICT = True

IMAGESAVE = False

MODEL = load_model("bestmodel.h5")

LABELS = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four",
          5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

pygame.init()
FONT = pygame.font.SysFont("freesansbold.tff", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWNSIZEX, WINDOWNSIZEY))

pygame.display.set_caption("Black Board")

iswriting = False

number_xcord = []
number_ycord = []

while True:  
    for event in pygame.event.get():  
        if event.type == QUIT:  
            pygame.quit()  
            sys.exit()  

        if event.type == MOUSEMOTION and iswriting:  
            xcord, ycord = event.pos  # Get the x and y coordinates of the mouse.
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            # Draw a white filled circle at the mouse's current position with a radius of 4 pixels

            number_xcord.append(xcord)  # Append the x coordinate to the list
            number_ycord.append(ycord)  # Append the y coordinate to the list

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and isinstance(number_ycord, list) and len(number_ycord) > 0:
                rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(number_xcord[-1] + BOUNDARYINC, WINDOWNSIZEX)
                rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(number_ycord[-1] + BOUNDARYINC, WINDOWNSIZEY)
            else:
                rect_min_x, rect_max_x, rect_min_y, rect_max_y = 0, 0, 0, 0  # Handle cases where data is missing or not iterable

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                image_cnt += 1 

            if PREDICT:
                if img_arr.shape[0] > 0 and img_arr.shape[1] > 0:
                    # Resize img_arr to (28, 28) and reshape to (1, 28, 28, 1)
                    image = cv2.resize(img_arr, (28, 28))
                    image = image / 255.0  # Normalize to the range [0, 1]
                    image = image.reshape(1, 28, 28, 1)  # Reshape to include a single grayscale channel

                    label = str(LABELS[np.argmax(MODEL.predict(image))])
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
