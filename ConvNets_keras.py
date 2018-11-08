''' Starter code for Convolutional Networks with Tensorflow and Keras
author = Boris P.
'''
import keras as K
import matplotlib as pyplot
import pickle


K.set_image_dim_ordering('th')

# load training data
pickle_in = open("TrainNC.pickle", "rb")
Train = pickle.load(pickle_in)
pickle_lbs = open("TrainNC_Lbs.pickle", "rb")
labels_Tr = pickle.load(pickle_lbs)

# load validation data
''' your code here'''

# load test data
''' your code here'''

# Visualize the first 9 samples
test_new_img_plt = Train.reshape(10000, 784)
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(test_new_img_plt[i+9].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
pyplot.show()

# Preprossesing data for training
# reshape to be [samples][pixels][width][height]
Train = Train.reshape(X_test1.shape[0], 1, 28, 28).astype('float32')
# Do the same for the validation and test data
''' your code here '''

# normalize inputs from 0-255 to 0-1
Train =
Validation =
Test =
# one hot encode outputs
y_train = np_utils.to_categorical(labels_Tr)
y_validation =
y_test =
num_classes = y_test.shape[1]

# define the larger model
def larger_model():
	# create model
	model = Sequential()
    '''complete the structure'''
    # convolution
	model.add()
    # maxPooling
	model.add()
    # convolution
	model.add()
    # maxPooling
	model.add()
    # Dropout
	model.add()
	model.add(Flatten()) # leave this as is.
    # Fully connected 128
	model.add()
    # Fully connected 50
	model.add()
    # Softmax
	model.add()
	# Compile model. Use cross entropy as loss function and the Adam optimizer!
	model.compile(... , metrics=['accuracy'])
	return model

# build the model
model = larger_model()
# Fit the model
model.fit(Train, y_train, validation_data=(Validation, y_validation), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(Test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))