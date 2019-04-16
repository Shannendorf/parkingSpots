from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
import os


SAVE_IMAGES = True # Set to True if you want to save the images that the program cuts out


def prepare(img_array):
    IMG_SIZE = 50
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# For testing purposes, print the location of the mouse when you leftclick on the video
def printLocation(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("{}, {}".format(x,y))


# Make necissary direcotries
videoDir = "Video-opnames Smart Parking"
if not os.path.isdir(videoDir):
    os.mkdir(videoDir)

imageOutDirFull = "output images (Full)"
if not os.path.isdir(imageOutDirFull):
    os.mkdir(imageOutDirFull)

imageOutDirEmpty = "output images (Empty)"
if not os.path.isdir(imageOutDirEmpty):
    os.mkdir(imageOutDirEmpty)

# Car locations y1,y2,x1,x2
cL = []

with open("coords.txt", "r") as f:
    coords = f.readlines()
    for coord in coords:
        cL.append(list(map(int, coord.strip().split(','))))


CATEGORIES = ["free", "busy"]
model = tf.keras.models.load_model('64x3-10epoch-new-CNN.model')


# #Capture from video feed:
# cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture('Video-opnames Smart Parking/1553861675359.mp4') # alles vol
# cap = cv2.VideoCapture('Video-opnames Smart Parking/1553880837766.mp4') # Bijna leeg, 1 man vertrekt
cap = cv2.VideoCapture('Video-opnames Smart Parking/1553877508109.mp4') # Zichtbare problemen


fps = cap.get(cv2.CAP_PROP_FPS) # Get fps of camera/video
secondsPerInterval = 6
interval = int(fps * secondsPerInterval) # Hiermee verander je de interval van hoe lang het duurt om footage te proberen


# Array om mee bij te houden welke plaatsen bezet zijn
pSpace = []
for car in cL:
    pSpace.append("")


intervalCount = 0
framecount = 0
first = True


while(cap.isOpened()):
    ### Capture frame-by-frame
    ret, frame = cap.read()

    # Als het lezen van de frame is gelukt:
    if ret==True:

        # Only do this the first time
        if first:
            height, width = frame.shape[:2]
            print("Height: {}  Width: {}".format(height, width))
            print("Fps: {}".format(int(fps)))
            print("The interval is {} seconds".format(int(interval/fps)))
            first = False

        ### Our operations on the frame come here

        # Only do this every interval
        if (framecount % interval) == 0:
            print("{} seconds of video footage has gone by".format(int(interval/fps)))
            #Voor alle aangegeven auto locaties
            for c in range(len(cL)):
                # Cut out a specific part of the image
                car = frame[cL[c][0]:cL[c][1], cL[c][2]:cL[c][3]]

                # Check if it has a car in it
                prediction = model.predict(prepare(car))
                pSpace[c] = CATEGORIES[int(prediction[0][0])]

                if SAVE_IMAGES == True:
                    # Save images to a file as jpg, you can change to png by changing the string to ".png"
                    if CATEGORIES[int(prediction[0][0])] == "free":
                        cv2.imwrite(os.path.join(imageOutDirEmpty,"location-{} interval-{}.jpg".format(c, intervalCount)), car)
                    else:
                        cv2.imwrite(os.path.join(imageOutDirFull, "location-{} interval-{}.jpg".format(c, intervalCount)), car)

            intervalCount += 1

        # Hier tekenen we nog wat shit op het scherm voor iedere parkeerplaats
        for c in range(len(cL)):

            # Teken vierkant op de plaats waar wordt gekeken
            cv2.rectangle(frame, (cL[c][2], cL[c][0]), (cL[c][3], cL[c][1]), (0, 255, 0), 1)

            # Teken rondje op vrije plaatsen
            if pSpace[c] == "free":
                cv2.circle(frame, ( (cL[c][2] + cL[c][3])//2, (cL[c][0] + cL[c][1])//2 ), 7, (0, 0, 255), -1)

        framecount = framecount + 1

        # If watching video, uncomment this if you want to be able to click and see the location of your cursor
        # cv2.setMouseCallback("frame", printLocation)

        ### Display the resulting frame
        cv2.imshow('frame',frame)

        ### stop when user hits 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()