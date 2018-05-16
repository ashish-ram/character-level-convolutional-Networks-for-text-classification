# char level classification of text:
The machine learning model is expected to learn to classify a given text input to one of the 12 novels in the training text data. 

## Training Text data: 
The training data contains three files - xtrain.txt, ytrain.txt, xtest.txt. Each line in the xtrain.txt comes from one of the 12 novels which are encoded by a whole number (0 to 11) and places in the ytrain.txt file in right order. xtest.txt contains test data for which each line should be classified to one of the 12 classes. 

## Model: 

In this project, we will be using a character level classification of text. That means text data is treated as a stream of characters instead of words (which is normally the case). This method is proposed by Zhang et al., 2015 (http://arxiv.org/abs/1509.01626). The main logic behind this type of unconvensional treatment to text data is that, by considering text as a stream of characters, the model can learn some of the morphological structures with a language which is missed if we treat a text on word level. 

First, we select the set of characters to be used. We will 26 lowercase english characters, 10 digits, 35 special characters and white space. Therefore, in total 69 characters. We choose to select first 1014 characters of each line in the training text according to the research paper. 

The characters can be encoded into one-hot vectors first. Therefore, each line in the training data will become a 2 dimensional array, just like an image. Now, this can be passed to a CNN layers and then fully connected layers. Finally, we will add the last layer with softmax which will output the classification probabilities. Additionally, we can add an embedding layer between the input and CNN layers.  

## Model training: 

We split the training data in training and heldout data. The heldout data will be used as a validation set during model training. We need to make sure that the heldout data is selected randomly. We use Adam optimizer and categorical crossentropy as loss function. 

Initially the model is trained just for 15 epochs which reseult in about 59% accuracy on validation dataset. In the research paper, the author report results close to state of the art in classification task.

