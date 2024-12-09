# Emotion Analysis Dashboard

## Project Overview
Emotion Analysis Dashboard is a real-time emotion recognition system that combines facial emotion analysis using a dataset and convolutional neural network (CNN) and text sentiment analysis powered by a Hugging Face's Transformers library. The application is built with Streamlit for easy deployment and interactivity.

---

## Packages and Dependencies

| Package      | Version | Description                              |
|--------------|---------|------------------------------------------|
| TensorFlow   | 2.6+    | For training and testing the facial recognition model |
| Streamlit    | 1.1+    | For building the web app interface       |
| OpenCV       | 4.5+    | For reading and loading the FER2013 dataset |
| Transformers | 4.11+   | For sentiment analysis using Hugging Face |

### Installation
Install the necessary dependencies using pip:
```bash
pip install tensorflow streamlit opencv-python transformers
```
---

# How to Run the Project

## 1. Download the FER2013 Dataset
- Download the FER2013 dataset.
- Place the dataset into the `FER2013_dataset/` directory with the following structure:

```bash
FER2013_dataset/
├── train/
├── test/

---

## Features
- **Text Sentiment Analysis**: Type text in the provided input field to analyze the sentiment (e.g., Positive, Negative, Neutral).
- **Facial Emotion Recognition**: Turn on your webcam to recognize facial emotions in real time (e.g., Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).

---

## References
Below are the materials and resources used in developing this project:
- **Dataset**: [FER2013 Facial Emotion Recognition Dataset](#)
- **Tutorials and Resources**:
  - Streamlit Documentation
  - Hugging Face Tutorials
  - TensorFlow Tutorials
- **Libraries**:
  - OpenCV
  - TensorFlow
- **Blogs and Code Snippets**:
  - Blog article on facial emotion recognition using TensorFlow
  - Tips for integrating OpenCV with Streamlit
