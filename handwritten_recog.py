

import numpy as np
from tensorflow import keras
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
# from keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers import SGD,Adam


def prepare_dataset():
    
  
    # for training and testing the neural network, data from Kaggle "A-Z Handwritten Alphabets.csv  " 
    # (https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)  and
    #  "MNIST Handwritten Digit Classification Dataset "

  
    #loading data from MNIST
    (trainX_mnist, trainy_mnist), (testX_mnist, testy_mnist) = mnist.load_data()
    

    #loading data from  A_Z Handwritten Data.csv
    data_A_Z = pd.read_csv(r"A_Z Handwritten Data.csv").fillna(0).astype('int8')


    # Shuffle the DataFrame from A_Z Handwritten Data.csv
    data_A_Z = data_A_Z.sample(frac=1)
    X_data_A_Z = data_A_Z.drop('0',axis = 1)
    y_data_A_Z = data_A_Z['0']


    #prepare_DataFrame A_Z Handwritten Data.csv 
    for i in range (0,26): y_data_A_Z=y_data_A_Z.replace(i,i+65)
    for i in range(65,91): y_data_A_Z=y_data_A_Z.replace(i,i-55)
    trainX_data_A_Z, testX_data_A_Z, trainy_data_A_Z, testy_data_A_Z = train_test_split(X_data_A_Z,y_data_A_Z, test_size = 0.2)
    trainX_data_A_Z = np.reshape(trainX_data_A_Z.values, (trainX_data_A_Z.shape[0], 28,28))
    testX_data_A_Z = np.reshape(testX_data_A_Z.values, (testX_data_A_Z.shape[0], 28,28))


    #concatenate MNIST and A_Z Handwritten Data.csv
    TRAIN_X = np.concatenate((trainX_mnist, trainX_data_A_Z), axis=0)
    TRAIN_Y = np.concatenate((trainy_mnist, trainy_data_A_Z), axis=0)
    TEST_X = np.concatenate((testX_mnist, testX_data_A_Z), axis=0)
    TEST_Y = np.concatenate((testy_mnist, testy_data_A_Z), axis=0)


    ## reshape dataset to have a single channel
    TRAIN_X = TRAIN_X.reshape((TRAIN_X.shape[0], 28, 28, 1))
    TEST_X = TEST_X.reshape((TEST_X.shape[0], 28, 28, 1))




    # one hot encode target values
    TRAIN_Y = to_categorical(TRAIN_Y)
    TEST_Y = to_categorical(TEST_Y)
    


    ## scale pixels
    def prep_pixels(train, test):
      # convert from integers to floats
      train_norm = train.astype('float32')
      test_norm = test.astype('float32')
      # normalize to range 0-1
      train_norm = train_norm / 255.0
      test_norm = test_norm / 255.0
      # return normalized images
      return train_norm, test_norm

    TRAIN_X, TEST_X = prep_pixels(TRAIN_X, TEST_X)



    return TRAIN_X, TRAIN_Y, TEST_X, TEST_Y

def build_model(input_shape, num_classes):
    # define cnn model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(64,activation ="relu"))
    model.add(Dense(128,activation ="relu"))
    model.add(Dense(num_classes,activation ="softmax"))
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs): 
    # Compile the model
    model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs,  validation_data = (X_test,y_test))



def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")



def save_model(model):
    # Save the model
    model.save_weights('unet_model_w.h5')
    model.save('unet_model.h5')

def main():
    # Set hyperparameter
    epochs = 5

    # Prepare the dataset
    X_train, y_train, X_test, y_test = prepare_dataset()

    # Build the model
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_model(input_shape, num_classes)

    # Train the model
    train_model(model, X_train, y_train, X_test, y_test, epochs)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, "handwritten_model.h5")



if __name__ == '__main__':
    main()

