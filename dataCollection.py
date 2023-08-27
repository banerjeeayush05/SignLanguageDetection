import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import uuid
import os

#Creating the OpenCV camera object
cap = cv2.VideoCapture(0)
#Creating the HandDetector object
detector = HandDetector(maxHands = 1)

#Defining constants
OFFSET = 30
IMG_SIZE = 300

#Directory path for the training images
folder_paths = ["A", "B", "C"]

#Main script
if(__name__ == "__main__"):

    #Creating a Data directory
    if(not os.path.exists("Data")):
        os.mkdir("Data")

    #Getting user input for which folder images will be added to
    folder_index = -1
    while True:
        folder_index = int(input("Folder index? 0, 1, 2: "))
        if(0 <= folder_index and folder_index <= len(folder_paths) - 1):
            break
        else:
            print("Please enter a valid number.")
            continue
    
    if(not os.path.exists(f"Data/{folder_paths[folder_index]}")):
        os.mkdir(f"Data/{folder_paths[folder_index]}")

    '''Getting user input if images want to be added with the previous images 
    or starting from no images in the specified folder'''
    add_images = ""
    while True:
        add_images = input("Add images to folder? T/F: ")

        if add_images.capitalize() == "T" or add_images.capitalize() == "True":
            add_images = True
            break
        elif add_images.capitalize() == "F" or add_images.capitalize() == "False":
            add_images = False
            break
        else:
            print('Please enter True or False')
            continue
    
    #Camera loop
    while cap.read():
        #Extracting image and implement hand object detection model
        success, img = cap.read()
        hands, img = detector.findHands(img)

        #Counting number of images in the specified image folder
        count = 0
        for path in os.listdir(f"Data/{folder_paths[folder_index]}"):
        # check if current path is a file
            if os.path.isfile(os.path.join(f"Data/{folder_paths[folder_index]}", path)):
                count += 1

        #Displaying the image count on the OpenCV camera image feed
        cv2.putText(img, f"{count} images in {folder_paths[folder_index]}", (0, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        
        if hands:
            #Getting the bounding box information of the hand and overlaying the image onto a new image
            hand_x, hand_y, hand_width, hand_height = hands[0]["bbox"]
            imgCrop = img[hand_y - OFFSET: hand_y + hand_height + OFFSET, hand_x - OFFSET: hand_x + hand_width + OFFSET]
            imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

            #Resizing the new image to 300x300 using the aspect ratio
            aspect_ratio = hand_height / hand_width
            if(aspect_ratio > 1):
                height_scale_factor = IMG_SIZE / hand_height
                new_hand_width = math.ceil(height_scale_factor * hand_width)
                width_gap = math.ceil((IMG_SIZE - new_hand_width) / 2)

                imgResize = cv2.resize(imgCrop, (new_hand_width, IMG_SIZE))
                imgWhite[:, width_gap: new_hand_width + width_gap] = imgResize
            else:
                width_scale_factor = IMG_SIZE / hand_width
                new_hand_height = math.ceil(width_scale_factor * hand_height)
                height_gap = math.ceil((IMG_SIZE - new_hand_height) / 2)

                imgResize = cv2.resize(imgCrop, (IMG_SIZE, new_hand_height))
                imgWhite[height_gap: new_hand_height + height_gap, :] = imgResize

        #Showing the OpenCV camera image feed
        cv2.imshow("Image", img)

        #Processing key inputs
        key = cv2.waitKey(1)

        #Removing all images from specified directory if add_images is False
        if(key == ord("s")):
            if(not add_images):
                add_images = True
                for image in os.listdir(f"Data/{folder_paths[folder_index]}"):
                    os.remove(os.path.join(f"Data/{folder_paths[folder_index]}", image))

            #Writing the processed hand image to the specified folder
            cv2.imwrite(f"Data/{folder_paths[folder_index]}/Image_{str(uuid.uuid1())}.jpg", imgWhite)

        #Killing the camera loop
        elif(key == ord("q")):
            print(f"Data Collected for folder Data/{folder_paths[folder_index]}")
            break
        
    #Properly destroying the OpenCV image feed
    cap.release()
    cv2.destroyAllWindows()