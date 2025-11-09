# Bell Pepper Health Classifier (ML Project)
dataset link : https://www.kaggle.com/datasets/arjuntejaswi/plant-village

This project trains a deep learning model to classify bell pepper images as **healthy** or **unhealthy** using transfer learning (ResNet50) and TensorFlow/Keras.

## Features

- Loads images and labels from Google Drive
- Preprocessing with OpenCV and NumPy
- Model is based on pre-trained ResNet50 (ImageNet)
- Training, validation, and test splits
- Early stopping and checkpointing
- Evaluation metrics: accuracy, confusion matrix, ROC AUC
- Grad-CAM visualizations
- Model export to HDF5 and TensorFlow.js formats

## How to Run

1. Download `bell_pepper_classifier.py` (or notebook) from this repo.
2. Make sure you have the required dependencies installed:
   - tensorflow
   - scikit-learn
   - opencv-python
   - matplotlib
   - numpy
3. Adjust file and dataset paths as needed.
4. Run:


For Jupyter/Colab:
- Open the notebook and run cells step by step.

## Dataset

- Expects images organized in Google Drive:

## Outputs

- Trained model weights and history
- Metrics and plots
- Saved model files (HDF5, TensorFlow.js)

## License

For educational and research purposes.

---

*Replace "bell_pepper_classifier.py" with the actual filename as appropriate.*

