# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
import os
import shutil

path1 = "F:\Machine Learning Projects\Malaria Detection\cell_images\Parasitized"
path2 = "F:\\Machine Learning Projects\\Malaria Detection\\cell_images\\Uninfected"

output1 = 'F:\Machine Learning Projects\Malaria Detection\data\\train\Parasitized'
output2 = 'F:\Machine Learning Projects\Malaria Detection\data\\test\Parasitized'
output3 = 'F:\\Machine Learning Projects\\Malaria Detection\\data\\train\\Uninfected'
output4 = 'F:\\Machine Learning Projects\\Malaria Detection\\data\\test\\Uninfected'

total_files1 = os.listdir(path1)
total_files2 = os.listdir(path2)

train_files1 = total_files1[:int(len(total_files1)*.8)]
test_files1 = total_files1[int(len(total_files1)*.8):]

train_files2 = total_files2[:int(len(total_files2)*.8)]
test_files2 = total_files2[int(len(total_files2)*.8):]

#train set for parasitized cells
for file in train_files1:
    #print(path1+'\\'+file, output1+'\\'+file)
    shutil.copy(path1+'\\'+file, output1+'\\'+file)
    
#test set for parasitized cells
for file in test_files1:
    #print(path1+'\\'+file, output2+'\\'+file)
    shutil.copy(path1+'\\'+file, output2+'\\'+file)

#training set for uninfected cells
for file in train_files2:
    #print(path2+'\\'+file, output3+'\\'+file)
    shutil.copy(path2+'\\'+file, output3+'\\'+file)

for file in test_files2:
    #print(path2+'\\'+file, output4+'\\'+file)
    shutil.copy(path2+'\\'+file, output4+'\\'+file)

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('F:\\Machine Learning Projects\\Malaria Detection\\data\\train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('F:\\Machine Learning Projects\\Malaria Detection\\data\\test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 22048,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 5510)

#saving the model
import h5py

classifier.save('F:/Machine Learning Projects/Malaria Detection/malaria_detection.h5')

from keras.models import load_model
new_model = load_model('F:/Machine Learning Projects/Malaria Detection/malaria_detection.h5')

new_model.summary()

new_model.get_weights()

new_model.optimizer

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('F:\\Machine Learning Projects\\Malaria Detection\\un.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = new_model.predict(test_image)
#new_model.class_indices
if result[0][0] == 1:
    prediction = 'Parasitized'
else:
    prediction = 'Uninfected'