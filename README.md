Data set Collection<br>
The Dataset was prepared after taking the data from Robo flow website. Dataset 
consists of Training and testing images. Training set consists of 200 Resume and 200
Non-Resume Images. Testing set consists of 50 Resume and 50 Non-Resume Images

Data Augumentation and Preprocessing<br>
Various techniques were performed such as centering the pixel values of each 
channel, randomly rotating and shifting the image horizontally, randomly flipping,
zooming and applying shear transformations for the images. The dimensions of each 
image was resized and batch size was set.

Model Selection<br>
Transfer Learning was used by loading Inception V3 model(Convolutional Neural 
Network) with weights pre trained on ImageNet. Flatten and Dense Layer was added 
on top for binary classification and model was compiled.

Model Training<br>
The model was trained using training data for 10 epochs while validating on test data. 
For training 100 training samples, and each batch contains 16 samples is processed 
per epoch. For validation 24 training samples, and each batch contains 16 samples is 
processed per epoch.

Testing and Confusion matrix<br>
Model.predict function was used to obtain predictions from the model on test 
dataset. A confusion matrix for predicted values and true values was used to get the 
accuracy of our model.
