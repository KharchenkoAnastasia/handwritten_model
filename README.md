# Handwritten Character Recognition

### **Description**


This project focuses on training and utilizing a neural network to classify squared black&white image (VIN character boxes) with single handwritten character on it. The project includes a training script to prepare, train, and save the model, as well as a test inference script that takes a directory path as a command-line argument and outputs the results in CSV format.

- **handwritten_model.h5** - binary file that stores model
- **handwritten_recog.py** - python file that prepare, train and save model
- **inference_script.py** - python file that takes one CLI argument that is path to directory with image samples and print output to console in CSV format
- **requarements.txt** - file  that prepares the environment


### **Prerequisites**
- Python 3.x
- TensorFlow
- Keras
- NumPy
- PIL (Python Imaging Library)



### **Installation**

#### Install the required dependencies:


```bash
pip install -r requirements.txt 
```


### **inference_script.py**

The inference_script.py takes a directory path as a command-line argument and performs inference on all images in that directory. The output is printed to the console in CSV format, displaying the ASCII index and POSIX path of each image sample.

1. Open a terminal or command prompt on your computer.
2. Navigate to the directory where the "inference_script.py" file
3. Replace <directory_path> in the command with the actual path to the directory containing your image samples. Make sure to provide the full path or relative path depending on your file system. 

```bash
python inference_script.py <directory_path>
```

### **handwritten_recog.py**
The handwritten_recog.py prepares, trains, and saves the neural network model. It is responsible for designing the architecture and training the model using the dataset.


To run the training script, use the following command:

```bash
python handwritten_recog.py
```


Ensure that the training dataset is properly configured and accessible within the script. You may need to adjust the script parameters, such as the number of epochs or batch size, to suit your specific requirements.

#### **Dataset**

For training and testing the neural network, data were taken from Kaggle "A-Z Handwritten Alphabets " (https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)  in .csv format and "MNIST Handwritten Digit Classification Dataset (from tensorflow.keras.datasets import mnist)".

 - The A-Z Handwritten Alphabets.csv dataset  contains 372450 images containing handwritten images in size 2828 pixels, each alphabet in the image is centre fitted to 20*20 pixel box.
- The MNIST dataset contains 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

MNIST and A-Z Handwritten Alphabets.csv  have been concatenated into one.

All labels are represented as floating point values in the dataset, have been converted to integer values, and a label dictionary has been created to map integer values to characters.

```python
label={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',\
           10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',\
               20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',\
         29:'T',30:'U',31:'V',32:'W',33:'X', 34:'Y',35:'Z'}
```
#### **Design the Model**
Convolutional neural network (CNN) architecture was used to build a neural network model for classify squared black & white image with single handwritten character on it.

The overview of the model's architecture and the number of parameters associated with each layer:

```python
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
```


#### **Train the Neural Network**

```python
    # Compile the model
    model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    history = model.fit(X_train, y_train, epochs=5,  validation_data = (X_test,y_test))
```

#### **Result**
![acc](https://github.com/KharchenkoAnastasia/handwritten_model/assets/47922202/4f2eb537-3968-4063-b81a-daf7cbca6099)


**Test loss:** 0.0425

**Test accuracy:** 0.9882


### **Examples**

Run the test inference script:

```bash
python inference_script.py images
```
Output :

```bash
67, C:\Users\kharc\Desktop\handwritten_recog\images\letter_C.jpg
49, C:\Users\kharc\Desktop\handwritten_recog\images\letter_T.jpg
67, C:\Users\kharc\Desktop\handwritten_recog\images\number_1.png
50, C:\Users\kharc\Desktop\handwritten_recog\images\number_2.png
57, C:\Users\kharc\Desktop\handwritten_recog\images\number_5.png
```

#### **Contact**
For any inquiries or support, please contact kharchenko.13.08@gmail.com Anastasiia Kharchenko .



