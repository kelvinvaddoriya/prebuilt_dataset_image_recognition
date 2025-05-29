# Image Recognition with Prebuilt Dataset

This project demonstrates image recognition using a **pretrained ResNet50 model** on a prebuilt dataset (ImageNet). The tool provides a simple graphical interface for users to upload images and receive predictions with confidence scores.

## Features

- Upload an image for recognition.
- Predict image class using the **pretrained ResNet50 model**.
- Display image with predicted labels and probabilities.
- Easy-to-use **Tkinter** interface for image selection.
- Grayscale conversion via MATLAB integration.

## Requirements

- Python 3.8+
- MATLAB with `matlab.engine` for Python
- Python libraries:
  - `tkinter`
  - `matplotlib`
  - `PIL`
  - `numpy`
  - `tensorflow`
  - `matlab.engine`

Install dependencies:
```bash
pip install matplotlib pillow tensorflow matlab.engine
````

## Usage

1. Run the application:

   ```bash
   python image_recognition.py
   ```

2. The GUI will open. Click **Browse Image** to select an image for recognition.

3. The tool will:

   * Convert the selected image to grayscale using MATLAB.
   * Predict the image class using the **pretrained ResNet50 model**.
   * Annotate the image with the predicted label and confidence.

4. The annotated image will be displayed using Matplotlib.

## File Structure

```
.
├── image_recognition.py       # Main Python script
├── convert_to_grayscale.m     # MATLAB script for grayscale conversion
└── README.md                  # Project documentation
```

## How It Works

* **Grayscale Conversion**: The image is passed to MATLAB for conversion to grayscale.
* **Prediction**: The grayscale image is processed by the pretrained ResNet50 model (on ImageNet) for prediction.
* **Annotation**: The image is annotated with the predicted label and confidence score.

## Example Workflow

1. Select an image via the GUI.
2. Image is converted to grayscale.
3. Pretrained ResNet50 model predicts the class.
4. The result is displayed with predictions and probabilities.

## Pretrained Model Details

* **Model**: ResNet50 (from Keras applications)
* **Dataset**: ImageNet (1,000 classes)
* **Input Size**: 224x224 pixels

## Future Enhancements

* Allow selection of different pretrained models.
* Improve UI for enhanced usability.
* Support batch image predictions.

## References

* [ResNet-50 Architecture](https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758)
* [TensorFlow](https://www.tensorflow.org/)
* [Matplotlib](https://pypi.org/project/matplotlib/)
* [NumPy](https://pypi.org/project/numpy/)
