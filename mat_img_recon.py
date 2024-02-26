import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import matlab.engine

# Start MATLAB Engine
eng = matlab.engine.start_matlab()

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def recognize_and_annotate(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_orig = mpimg.imread(image_path)

    # Convert the image to RGB mode
    img = img.convert('RGB')

    # image recognition
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Annotate the image with prediction labels and probabilities
    plt.imshow(img_orig)
    plt.axis('off')

    plt.gcf().text(0.5, 0.02, "Predictions:", fontsize=12, ha='center')
    for i, pred in enumerate(decoded_predictions):
        probability = pred[2] * 100  # Convert probability to percentage
        plt.gcf().text(0.5, 0.02 + (i+1)*0.05, f"{pred[1]}: {probability:.2f}%", fontsize=10, ha='center')
    plt.show()

# Function to handle button click event
def browse_file():
    filename = filedialog.askopenfilename()
    if filename:
        # Call MATLAB function to convert image to grayscale
        grayscale_image_path = convert_to_grayscale(filename)
        # Perform image recognition and annotation on the grayscale image
        recognize_and_annotate(grayscale_image_path)

# MATLAB function to convert image to grayscale
def convert_to_grayscale(image_path):
    grayscale_image_path = eng.convert_to_grayscale(image_path, nargout=1)
    return grayscale_image_path

root = tk.Tk()
root.title("Image Recognition")

# browse and upload an image
browse_button = tk.Button(root, text="Browse Image", command=browse_file)
browse_button.pack()

root.mainloop()

# Stop MATLAB Engine
eng.quit()