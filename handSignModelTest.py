import cv2
from cvzone.HandTrackingModule import HandDetector
from torchvision import transforms
import torch
from handSignModelTrain import HandSignTinyVGGModel
import numpy as np
import math

#Defining predict function to return the predicted letter with the confidence
def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predictions = torch.softmax(predictions, dim = 1, dtype = torch.float32)
        predicted_index = torch.argmax(predictions)
        confidence = torch.max(predictions)
        predicted_value = class_mapping[predicted_index]
    return predicted_value, confidence

#Defining truncate function to display confidence values properly
def truncate(float, num_places):
    return math.floor(float * 10 ** num_places) / 10 ** num_places

#Indexes to classes array
class_mapping = ["A", "B", "C"]

#Creating the OpenCV camera object
cap = cv2.VideoCapture(0)
#Creating the HandDetector object
detector = HandDetector(maxHands = 1)

#Defining constants
OFFSET = 20
IMG_SIZE = 300
TRANSFORM = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size = (64, 64)), transforms.ToTensor()])

#Initializing dynamic variables
counter = 0
imgWhite = None

#Main script
if(__name__ == '__main__'):
    #Load back the model
    hand_sign_model = HandSignTinyVGGModel(input_shape = 3, hidden_units = 10, output_shape = 3)
    state_dict = torch.load("hand_sign_tiny_vgg_model.pth")
    hand_sign_model.load_state_dict(state_dict)

    #Camera loop
    while cap.read():
        #Extracting image and implement hand object detection model
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        #Pre-processing the camera image to a 300x300 image of the hand
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
            
            #Making an inference by converting the image to a tensor for the pytorch model
            imgWhiteTensor = TRANSFORM(imgWhite)
            #Reshaping the tensor from (3, 64, 64) to (1, 3, 64, 64)
            imgWhiteTensor = torch.reshape(imgWhiteTensor, (1, 3, 64, 64))
            predicted_letter, confidence = predict(hand_sign_model, imgWhiteTensor, class_mapping)

            #Displaying the predicted class of the hand along with the hand bounding box on the OpenCV camera image feed
            cv2.rectangle(imgOutput, (math.ceil(hand_x + hand_width / 2) - 70, hand_y - 60), (math.ceil(hand_x + hand_width / 2) + 70, hand_y - 20), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, f"{predicted_letter}: {truncate(float(confidence), 2)}", (math.ceil(hand_x + hand_width / 2) - 60, hand_y - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 4)
            cv2.rectangle(imgOutput, (hand_x - 20, hand_y - 20), (hand_x + hand_width + 20, hand_y + hand_height + 20), (255, 0, 255), 4)

        #Showing the OpenCV camera image feed
        cv2.imshow("Image", imgOutput)

        #Processing key inputs
        key = cv2.waitKey(1)
        #Killing the camera loop
        if(key == ord("q")):
            break
    
    #Properly destroying the OpenCV image feed
    cap.release()
    cv2.destroyAllWindows()