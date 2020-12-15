# Visual Question Answering (VQA)

A simple implementation of a model used to solve a VQA task. 
The model is restricted to only answer 'yes/no' questions.
The implementation is not meant to be competitive, 
it is part of a Deep Learning project given to students as an excellent
to combine their knowledge on NLP and Image processing using deep learning.

# List of used packages

- Pytorch v 1.7
- torchvision
- Huggingface transformers
- Pandas

# Model description 

The model is made of 3 parts : (i) an image feature extraction part that consists
of resnet18 pretrained model, (ii) an BERT model for encoding the words in the question 
followed by a LSTM cell to encode the question, (iii) a MLP classifier with 2 hidden linear layers
and relu activation function and a dropout layer in between. 

The image and questions representations are passed 
into 2 linear layers (one for each representation) to project the representations to a space with the same dimentions.
The 2 projections are then concatenated and passed to the classifier.


