# RescueAI - AI Emergency Trigger

RescueAI is a real-time emergency analysis web application that listens to audio input, transcribes it into text, and classifies the emergency into a Risk Level (Low, Medium, High). If a critical "High Risk" emergency is detected, it automatically prompts the user to dial 911 immediately.

This system is fine-tuned on real-world 911 dispatch datasets to accurately identify the severity of emergency situations based on spoken keywords and context.

## 🌟 Key Features

* **Real-time Audio Processing:** Uses the microphone directly from the browser to capture emergency statements.
* **Speech-to-Text Transcription:** Powered by OpenAI's Whisper model (`openai/whisper-small`) to convert spoken audio into accurate text.
* **Emergency Risk Classification:** Powered by a customized `DistilBERT` neural network fine-tuned on over 600 sampled real-world 911 emergency dispatch calls.
* **Dynamic Risk Alerts:** A sleek, glassmorphic UI that visually pulses red and displays a "CALL 911" alert when critical scenarios (like cardiac arrest or severe injuries) are detected.

## 🛠️ Technology Stack

* **Backend:** Python, FastAPI, Uvicorn
* **AI / Machine Learning:** PyTorch, Hugging Face `transformers`, `librosa`, Scikit-Learn
* **Frontend:** HTML5, Vanilla JavaScript, CSS3 (Glassmorphism design)
* **Datasets:** Kaggle 911 Emergency Dataset

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.9+ and `ffmpeg` installed on your system. 
*(Note: The app automatically attempts to load the `imageio_ffmpeg` binary for seamless audio conversion without requiring manual ffmpeg system installations!)*

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arbazkhan-ai/ai-emergency-trigger.git
   cd ai-emergency-trigger
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the AI model (Optional):**
   The application requires a fine-tuned `emergency_model`. If you have the `911.csv` dataset, you can map and train the DistilBERT model by running:
   ```bash
   python prepare_911_data.py
   python bert_train.py
   ```

### Running the Application

Start the FastAPI server:
```bash
python -m uvicorn app:app --reload
```

Then, open your web browser and navigate to:
**http://127.0.0.1:8000**

Click the microphone button, grant audio permissions, and describe an emergency to see the AI analysis in action!

## 📄 License
This project is open-source and available under the MIT License.
