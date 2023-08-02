# SignLanguageDetection

##Introduction
This is an OpenCV project that uses a PyTorch image classification model following the TinyVGG architecture to distinguish between different sign language hand symbols. Current model provided is trained to recognize between A, B, and C. With the current structure of the code, the model can be easily retrained to recognize more images. The main libraries you need are CVZone, OpenCV, and PyTorch.

You can learn more about the TinyVGG architecture [here](https://poloclub.github.io/cnn-explainer/).

##Data Collection
To collect the data, please use the dataCollection.py. To write images to the folder, put your hand up in the right sign language gesture and press the 's' key. Try to collect around 300 images per class to get a working model. The current model is trained with 320 images per class. 

##Training the Model
To train the model, use the handSignModelTrain.py. Fill free to tweak the model class and training hyperparameters to get better results. The model is then saved as hand_sign_tiny_vgg_model.pth. Every time you run train, the saved model is rewritten over the previous model.

##Testing the Model in Realtime
To test the model, please run the handSignModelTrain.py. This script gives the predicted class and the confidence level displayed on the screen.

##Future plans
Still some bugs to fix in the OpenCV camera closing when the bounding box leaves the screen. I also might expand this model to all 26 characters. I will add a requirements.txt file to make installing the packages easy and quick.
