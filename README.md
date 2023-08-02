# SignLanguageDetection

OpenCV project that uses a PyTorch image classification model following the TinyVGG architecture to distinguish between different sign language hand symbols. Current model provided is trained to recognize between A, B, and C. With the current structure of the code, the model can be easily retrained to recognize more images.

You can learn more about the TinyVGG architecture [here] (https://poloclub.github.io/cnn-explainer/)

To collect the data, please use the dataCollection.py. To write images to the folder, put your hand up in the right sign language gesture and press the 's' key. Try to collect around 300 images per class to get a working model. The current model is trained with 320 images per class.

To train the model, use the handSignModelTrain.py. 
To test the model, please run the handSignModelTrain.py. 

Still some bugs to fix in the OpenCV camera closing when the bounding box leaves the screen.
