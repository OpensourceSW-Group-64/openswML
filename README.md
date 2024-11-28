Emotion Analysis Dashboard
📖 Project Overview
Emotion Analysis Dashboard is a Streamlit-based application that analyzes emotions from facial expressions and text. It uses OpenCV and TensorFlow for real-time facial emotion recognition and Hugging Face Transformers for text sentiment analysis.

📦 Installation
Install required libraries:

pip install streamlit tensorflow opencv-python numpy transformers

Prepare the FER2013 dataset:
Download the FER2013 dataset and organize it as follows:
FER2013_dataset/
├── train/
└── test/

🚀 How to Run
Start the Streamlit app:
streamlit run emotion_analysis.py
Features:
Text Sentiment Analysis: Analyze emotions by entering text into the input field.
Facial Emotion Recognition: Use your webcam to analyze facial emotions in real-time.

🛠 Technologies Used
Facial Emotion Recognition: CNN model to classify emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).
Text Sentiment Analysis: Hugging Face’s distilbert-base-uncased-finetuned-sst-2-english model.

📂 File Structure
emotion-analysis-dashboard/
├── emotion_analysis.py   # Main application code
└── FER2013_dataset/      # Dataset
