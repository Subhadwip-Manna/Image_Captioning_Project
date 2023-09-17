# Image_Captioning_Project
![Screenshot 2023-09-13 111021](https://github.com/Subhadwip-Manna/Image_Captioning_Project/assets/140252649/18229137-9f66-4816-8aa0-37c11e99f6cf)  
![Screenshot 2023-09-13 111855](https://github.com/Subhadwip-Manna/Image_Captioning_Project/assets/140252649/862396a1-74c4-43d8-98e7-b4c6836f7bf8)


# What is Image Caption Generator?
Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English.

# Image Caption Generator with CNN
The objective of our project is to learn the concepts of a CNN and LSTM model and build a working model of Image caption generator by implementing CNN with LSTM.  
In this Python project, we will be implementing the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short term memory). The image features will be extracted from VGG16 which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.  
  
# What is CNN?
Convolutional Neural networks are specialized deep neural networks which can process the data that has input shape like a 2D matrix. Images are easily represented as a 2D matrix and CNN is very useful in working with images.

CNN is basically used for image classifications and identifying if an image is a bird, a plane or Superman, etc.  
It scans images from left to right and top to bottom to pull out important features from the image and combines the feature to classify images. It can handle the images that have been translated, rotated, scaled and changes in perspective.  

# What is LSTM?
LSTM stands for Long short term memory, they are a type of RNN (recurrent neural network) which is well suited for sequence prediction problems. Based on the previous text, we can predict what the next word will be. It has proven itself effective from the traditional RNN by overcoming the limitations of RNN which had short term memory. LSTM can carry out relevant information throughout the processing of inputs and with a forget gate, it discards non-relevant information.

# This is what an LSTM cell looks like –
![image](https://github.com/Subhadwip-Manna/Image_Captioning_Project/assets/140252649/f5a09e0a-f73c-4c24-baf1-47ed61941b52)  

# Image Caption Generator Model
So, to make our image caption generator model, we will be merging these architectures. It is also called a CNN-RNN model.

CNN is used for extracting features from the image. We will use the pre-trained model Xception.
LSTM will use the information from CNN to help generate a description of the image.
![image](https://github.com/Subhadwip-Manna/Image_Captioning_Project/assets/140252649/860c60fd-afc8-4fca-9806-a00dbf4743ac)  

# Project File Structure
You need to download from Kaggle:

Flicker8k_Dataset – Dataset folder which contains 8091 images.  
Flickr_8k_text – Dataset folder which contains text files and captions of images.



