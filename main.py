import numpy as np
import cv2
from matplotlib import pyplot as plt


# def printLocation(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print("{}, {}".format(x,y))


# #Capture from video feed:
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('Video-opnames Smart Parking/1553861675359.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)

#Every x seconds
interval = int(fps * 60)

IMGSIZE = (60,60)

#TODO Add additional parking locations
#Car locations y1,y2,x1,x2
cL = [[230, 295, 210, 270],
      [230, 295, 165, 225]]
      # [140, 290, 220, 270]]

f = 0
first = True

while(cap.isOpened()):
    ### Capture frame-by-frame
    ret, frame = cap.read()

    if ret==True:

        # Only do this the first time
        if first:
            height, width = frame.shape[:2]
            print("Height: {}  Width: {}".format(height, width))
            print("Fps: {}".format(int(fps)))
            print("The interval is {} seconds".format(int(interval/fps)))
            first = False


        ### Our operations on the frame come here

        ## Draw line
        # frame = cv2.line(frame,(0,0),(170,190),(255,0,0),1)

        # Only do this every interval
        if f == 0:
            print("{} seconds of video footage has gone by".format(int(interval/fps)))
            for c in cL:
                car = frame[c[0]:c[1], c[2]:c[3]]
                plt.imshow(car)
                plt.show()

        ## This could be an easier way for Shannon to go to grayscale?
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        f = (f + 1) % interval

        # cv2.setMouseCallback("frame", printLocation)

        ### Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()