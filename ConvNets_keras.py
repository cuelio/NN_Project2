''' Starter code for Convolutional Networks with Tensorflow and Keras
author = Boris P.
'''
from keras import backend as K
import matplotlib.pyplot as pyplot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import pickle

dataset_dir = "data/"

K.set_image_dim_ordering('th')

# load training data

with open(dataset_dir + "centered_data/TrainC.pickle", 'rb') as file_reader:
    centered_train_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "centered_data/TrainC_Lbs.pickle", 'rb') as file_reader:
    centered_train_data_labels = pickle.load(file_reader, encoding="latin1")

# load validation data

with open(dataset_dir + "centered_data/TestC.pickle", 'rb') as file_reader:
    centered_validation_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "centered_data/TestC_Lbs.pickle", 'rb') as file_reader:
    centered_validation_data_labels = pickle.load(file_reader, encoding="latin1")

# load test data

with open(dataset_dir + "centered_data/ValidationC.pickle", 'rb') as file_reader:
    centered_test_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "centered_data/ValidationC_Lbs.pickle", 'rb') as file_reader:
    centered_test_data_labels = pickle.load(file_reader, encoding="latin1")

# Visualize the first 9 samples
# print(centered_train_data.shape)
# test_new_img_plt = centered_train_data.reshape(55000, 784)
# for i in range(0, 9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(test_new_img_plt[i+9].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
# pyplot.show()

# Preprossesing data for training
# reshape to be [samples][pixels][width][height]
Train = centered_train_data.reshape(centered_train_data.shape[0], 1, 28, 28).astype('float32')
Test = centered_test_data.reshape(centered_test_data.shape[0], 1, 28, 28).astype('float32')
Validation = centered_validation_data.reshape(centered_validation_data.shape[0], 1, 28, 28).astype('float32')

# one hot encode outputs
y_train = np_utils.to_categorical(centered_train_data_labels)
y_validation = np_utils.to_categorical(centered_validation_data_labels)
y_test =np_utils.to_categorical(centered_test_data_labels)
num_classes = y_test.shape[1]

# define the larger model
def larger_model():
    # create model
    model = Sequential()
    # convolution
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # maxPooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # convolution
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # maxPooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model. Use cross entropy as loss function and the Adam optimizer!
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
print("Starting to test model")
model = larger_model()
# Fit the model
model.fit(Train, y_train, validation_data=(Validation, y_validation), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(Test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))