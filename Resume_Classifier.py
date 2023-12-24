#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Loading the InceptionV3 model with weights pre-trained on ImageNet.
#The model is configured to take input images of size 256x256 pixels with three color channels.
base_model = InceptionV3(input_shape=(256,256,3), include_top= False)


#setting all layers in the InceptionV3 base model to be non-trainable
for layer in base_model.layers:
  layer.trainable = False

#Custom layers (Flatten and Dense) are added on top of the InceptionV3 base model to create the final model.
#The output layer has two units with a sigmoid activation function, indicating a binary classification task.
X = Flatten()(base_model.output)
X = Dense(units=2, activation= 'sigmoid') (X)

# Final Model
model = Model(base_model.input, X)

# compile the model
model.compile(optimizer = 'adam', loss = keras.losses.binary_crossentropy, metrics=['accuracy'])

#summary
model.summary()

#Image data generators are set up for training with specified augmentation parameters.
#Training data is loaded from directory.
train_datagen = ImageDataGenerator(featurewise_center= True,rotation_range= 0.4,width_shift_range= 0.3,horizontal_flip= True,preprocessing_function= preprocess_input,zoom_range= 0.4,shear_range= 0.4)
train_data = train_datagen.flow_from_directory(directory= "/content/TrainDataSet", target_size=(256,256), batch_size= 36)

#Checking indices of training data
train_data.class_indices

#Image data generators are set up for testing with specified augmentation parameters.
#Testing data is loaded from directory.
test_datagen = ImageDataGenerator(featurewise_center= True,rotation_range= 0.4,width_shift_range= 0.3,horizontal_flip= True,preprocessing_function= preprocess_input,zoom_range= 0.4,shear_range= 0.4)
test_data = test_datagen.flow_from_directory(directory= "/content/TestDataSet", target_size=(256,256), batch_size= 36)

#Checking indices of testing data
test_data.class_indices


t_img , label = train_data.next()

t_img.shape

#To display the images and their shapes for the first 10 samples in the batch.
def plotImages(img_arr , label):
  for idx , img in enumerate(img_arr):
    if idx <=10 :
      plt.figure(figsize=(5,5))
      plt.imshow(img)
      plt.title(img.shape)
      plt.axis = False
      plt.show()

plotImages(t_img , label)

#The model is trained using the fit_generator method
#with some callback functions like ModelCheckpoint and EarlyStopping.
his = model.fit_generator(
   train_data,
    steps_per_epoch=100 // 16,
    epochs=10,
    validation_data=test_data,validation_steps=24 // 16)

#For h.keys()
h = his.history
h.keys()

#For plotting loss and accuracy
plt.plot(h['loss'])
plt.plot(h['accuracy'], 'go--', c = "red" , )
plt.title("Loss vs Acc")
plt.show()

#Plotting Accuracy
plt.plot(his.history['accuracy'])
plt.plot(his.history['val_accuracy'])

#The trained model is evaluated on the test data
predictions = model.predict(test_data)
binary_predictions = (predictions > 0.5).astype(int)

true_labels = test_data.classes

# A confusion matrix is calculated using scikit-learn's confusion_matrix function.
conf_matrix = confusion_matrix(true_labels, binary_predictions[:, 0])

#The confusion matrix is visualized using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()