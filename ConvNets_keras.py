''' Starter code for Convolutional Networks with Tensorflow and Keras
author = Boris P.
'''
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
from keras import backend as K
import matplotlib.pyplot as pyplot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import pickle


dataset_dir = "data/"

K.set_image_dim_ordering('th')

# CENTERED DATA

# load centered training data

with open(dataset_dir + "centered_data/TrainC.pickle", 'rb') as file_reader:
    centered_train_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "centered_data/TrainC_Lbs.pickle", 'rb') as file_reader:
    centered_train_data_labels = pickle.load(file_reader, encoding="latin1")

# load centered test data

with open(dataset_dir + "centered_data/TestC.pickle", 'rb') as file_reader:
    centered_validation_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "centered_data/TestC_Lbs.pickle", 'rb') as file_reader:
    centered_validation_data_labels = pickle.load(file_reader, encoding="latin1")

# load centered validation data

with open(dataset_dir + "centered_data/ValidationC.pickle", 'rb') as file_reader:
    centered_test_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "centered_data/ValidationC_Lbs.pickle", 'rb') as file_reader:
    centered_test_data_labels = pickle.load(file_reader, encoding="latin1")


# UNCENTERED DATA

# load uncentered training data

with open(dataset_dir + "uncentered_data/TrainNC.pickle", 'rb') as file_reader:
    uncentered_train_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "uncentered_data/TrainNC_Lbs.pickle", 'rb') as file_reader:
    uncentered_train_data_labels = pickle.load(file_reader, encoding="latin1")

# load uncentered test data

with open(dataset_dir + "uncentered_data/TestNC.pickle", 'rb') as file_reader:
    uncentered_validation_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "uncentered_data/TestNC_Lbs.pickle", 'rb') as file_reader:
    uncentered_validation_data_labels = pickle.load(file_reader, encoding="latin1")

# load uncentered validation data

with open(dataset_dir + "uncentered_data/ValidationNC.pickle", 'rb') as file_reader:
    uncentered_test_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "uncentered_data/ValidationNC_Lbs.pickle", 'rb') as file_reader:
    uncentered_test_data_labels = pickle.load(file_reader, encoding="latin1")


# POSITION AND SIZE INVARIANT

# load position and size invariant training data

with open(dataset_dir + "position_and_size_invariant/TrainNCS.pickle", 'rb') as file_reader:
    psi_train_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "position_and_size_invariant/TrainNCS_Lbs.pickle", 'rb') as file_reader:
    psi_train_data_labels = pickle.load(file_reader, encoding="latin1")

# load position and size invariant test data

with open(dataset_dir + "position_and_size_invariant/TestNCS.pickle", 'rb') as file_reader:
    psi_test_data = pickle.load(file_reader, encoding="latin1")

with open(dataset_dir + "position_and_size_invariant/TestNCS_Lbs.pickle", 'rb') as file_reader:
    psi_test_data_labels = pickle.load(file_reader, encoding="latin1")

# Visualize the first 9 samples
#
# print(centered_train_data.shape)
# test_new_img_plt = centered_train_data.reshape(55000, 784)
# for i in range(0, 9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(test_new_img_plt[i+9].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
# pyplot.show()

# Preprossesing data for training
# reshape to be [samples][pixels][width][height]
Train_centered = centered_train_data.reshape(centered_train_data.shape[0], 1, 28, 28).astype('float32')
Test_centered = centered_test_data.reshape(centered_test_data.shape[0], 1, 28, 28).astype('float32')
Validation_centered = centered_validation_data.reshape(centered_validation_data.shape[0], 1, 28, 28).astype('float32')
Train_uncentered = uncentered_train_data.reshape(uncentered_train_data.shape[0], 1, 28, 28).astype('float32')
Test_uncentered = uncentered_test_data.reshape(uncentered_test_data.shape[0], 1, 28, 28).astype('float32')
Validation_uncentered = uncentered_validation_data.reshape(uncentered_validation_data.shape[0], 1, 28, 28).astype('float32')
Train_psi = psi_train_data.reshape(psi_train_data.shape[0], 1, 28, 28).astype('float32')
Test_psi = psi_test_data.reshape(psi_test_data.shape[0], 1, 28, 28).astype('float32')
Train_centered_uncentered = np.concatenate((Train_centered, Train_uncentered), axis=0)
Test_centered_uncentered = np.concatenate((Test_centered, Test_uncentered), axis=0)
Validation_centered_uncentered = np.concatenate((Validation_centered, Validation_uncentered), axis=0)

# one hot encode outputs
y_train_centered = np_utils.to_categorical(centered_train_data_labels)
y_validation_centered = np_utils.to_categorical(centered_validation_data_labels)
y_test_centered = np_utils.to_categorical(centered_test_data_labels)
y_train_uncentered = np_utils.to_categorical(uncentered_train_data_labels)
y_validation_uncentered = np_utils.to_categorical(uncentered_validation_data_labels)
y_test_uncentered =np_utils.to_categorical(uncentered_test_data_labels)
y_train_psi = np_utils.to_categorical(psi_train_data_labels)
y_test_psi = np_utils.to_categorical(psi_test_data_labels)
y_train_centered_uncentered = np.concatenate((y_train_centered, y_train_uncentered), axis=0)
y_test_centered_uncentered = np.concatenate((y_test_centered, y_test_uncentered), axis=0)
y_validation_centered_uncentered = np.concatenate((y_validation_centered, y_validation_uncentered), axis=0)
num_classes = y_test_centered.shape[1]

# define the larger model
def part_bcd_conv_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def part_bcd_3_model():
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def competition_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Booleans for running different parts of the Project / Competition
run_bc_conv = True
run_bc_3 = False
run_d1_conv = False
run_d1_3 = False
run_d2_conv = False
run_d2_3 = False
competition = False

bc_conv_history = None
bc_3_history = None
d1_conv_history = None
d1_3_history = None
d2_conv_history = None
d2_3_history = None

# plots training loss and validation loss after each epoch
def plot_history(history):
    # Plot training & validation accuracy values
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('Model accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper left')
    pyplot.show()

    # Plot training & validation loss values
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Model loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper left')
    pyplot.show()

# Part B and C of the Project. Train, Test, and Evaluate on centered data
if run_bc_conv:
    # build the CNN model for part b and c of the project
    model = part_bcd_conv_model()
    bc_conv_history = model.fit(Train_centered, y_train_centered, validation_data=(Validation_centered, y_validation_centered), epochs=2, batch_size=200)
    scores = model.evaluate(Test_centered, y_test_centered, verbose=0)
    print("Centered Trained & Tested CNN Error: %.2f%%" % (100-scores[1]*100))
    plot_history(bc_conv_history)

if run_bc_3:
    # build the CNN model for part b and c of the project
    model = part_bcd_3_model()
    bc_3_history = model.fit(Train_centered, y_train_centered, validation_data=(Validation_centered, y_validation_centered), epochs=10, batch_size=200)
    scores = model.evaluate(Test_centered, y_test_centered, verbose=0)
    print("Centered Trained & Tested 3-Layer Network Error: %.2f%%" % (100-scores[1]*100))

# Part D subpart 1 of the Project. Train on centered data, Test on uncentered data
if run_d1_conv:
    # build the CNN model for part b and c of the project
    model = part_bcd_conv_model()
    d1_conv_history = model.fit(Train_uncentered, y_train_uncentered, validation_data=(Validation_uncentered, y_validation_uncentered), epochs=10, batch_size=200)
    scores = model.evaluate(Test_uncentered, y_test_uncentered, verbose=0)
    print("Centered Trained & Uncentered Tested CNN Error: %.2f%%" % (100-scores[1]*100))
if run_d1_3:
    # build the CNN model for part b and c of the project
    model = part_bcd_3_model()
    d1_3_history = model.fit(Train_uncentered, y_train_uncentered, validation_data=(Validation_uncentered, y_validation_uncentered), epochs=10, batch_size=200)
    scores = model.evaluate(Test_uncentered, y_test_uncentered, verbose=0)
    print("Centered Trained & Uncentered Tested 3-Layer Network Error: %.2f%%" % (100-scores[1]*100))

# Part D subpart 2 of the Project. Train on centered + uncentered data, Test on centered + uncentered data
if run_d2_conv:
    # build the CNN model for part b and c of the project
    model = part_bcd_conv_model()
    d2_conv_history = model.fit(Train_centered_uncentered, y_train_centered_uncentered, validation_data=(Validation_centered_uncentered, y_validation_centered_uncentered), epochs=10, batch_size=200)
    scores = model.evaluate(Test_centered_uncentered, y_test_centered_uncentered, verbose=0)
    print("Centered-Uncentered Trained & Centered-Uncentered Tested CNN Error: %.2f%%" % (100-scores[1]*100))
if run_d2_3:
    # build the CNN model for part b and c of the project
    model = part_bcd_3_model()
    d2_3_history = model.fit(Train_centered_uncentered, y_train_centered_uncentered, validation_data=(Validation_centered_uncentered, y_validation_centered_uncentered), epochs=10, batch_size=200)
    scores = model.evaluate(Test_centered_uncentered, y_test_centered_uncentered, verbose=0)
    print("Centered-Uncentered Trained & Centered-Uncentered Tested 3-Layer Network Error: %.2f%%" % (100-scores[1]*100))

# Part E / Competition of the Project.
if competition:
    model = competition_model()
    model.fit(Train_psi, y_train_psi, epochs=1, batch_size=200)
    scores = model.evaluate(Test_psi, y_test_psi, verbose=0)
    print("Centered Trained & Tested CNN Error: %.2f%%" % (100 - scores[1] * 100))