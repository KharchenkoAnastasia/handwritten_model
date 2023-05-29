import os
import sys
import csv
from tensorflow import keras
from PIL import Image
import numpy as np
from numpy import argmax

# Load the saved model
model = keras.models.load_model("handwritten_model.h5")

def perform_inference(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Adjust image size according to your model
    image_array = 1 - (np.array(image) / 255.0)  # Normalize and invert colors

    # Reshape the image to match the input shape of the model
    image_input = image_array.reshape((1, 28, 28, 1))

    # Perform inference
    label={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',\
           10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',\
               20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',\
         29:'T',30:'U',31:'V',32:'W',33:'X', 34:'Y',35:'Z'}
    predicted_label =label[argmax(model.predict(image_input)[0])]
    

    return ord(predicted_label)

def main(directory):
    # Find all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Perform inference on each image and print the results
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        predicted_label = perform_inference(image_path)
        result = f"{predicted_label}, {os.path.abspath(image_path)}"
        print(result)
        

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python inference_script.py <directory_path>")
        sys.exit(1)
        

    directory_path = sys.argv[1]
    main(directory_path)
    # main("images")
