from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import torch
import librosa
import tempfile
import os
import imageio_ffmpeg

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
ffmpeg_target = os.path.join(ffmpeg_dir, "ffmpeg.exe")
if not os.path.exists(ffmpeg_target):
    import shutil
    shutil.copyfile(ffmpeg_exe, ffmpeg_target)
os.environ["PATH"] += os.pathsep + ffmpeg_dir

app = FastAPI(title="Emergency AI Assistant")

# Mount static files for UI dashboard
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models
text_model = None
whisper_model = None
id2label = {
    0: "LOW RISK",
    1: "MEDIUM RISK",
    2: "HIGH RISK"
}

@app.on_event("startup")
async def load_models():
    global text_model, whisper_model
    device = 0 if torch.cuda.is_available() else -1
    
    print("Loading Text Classification Model...")
    text_model = pipeline(
        "text-classification",
        model="emergency_model",
        tokenizer="emergency_model",
        device=device,
        top_k=None
    )
    
    print("Loading Whisper ASR Model...")
    whisper_model = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device
    )
    print("Models loaded successfully!")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_audio(audio: UploadFile = File(...)):
    try:
        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        # Load audio using librosa at 16kHz (Whisper expected sample rate)
        # librosa can read webm if ffmpeg is installed
        speech_array, sampling_rate = librosa.load(tmp_path, sr=16000)
        
        # Clean up temp file
        os.remove(tmp_path)
        
        # 1. Transcribe audio to text
        print("Transcribing...")
        transcription_result = whisper_model(speech_array, generate_kwargs={"task": "transcribe"})
        transcribed_text = transcription_result["text"].strip()
        print(f"Transcribed Text: {transcribed_text}")
        
        if not transcribed_text:
            return JSONResponse({"error": "Could not transcribe audio."}, status_code=400)
        
        # 2. Classify text for emergency risk
        outputs = text_model(transcribed_text)
        scores = outputs[0] if isinstance(outputs[0], list) else outputs
        best = max(scores, key=lambda x: x["score"])
        
        label_id = int(best["label"].split("_")[-1])
        risk_level = id2label[label_id]
        confidence = best["score"]
        
        return {
            "text": transcribed_text,
            "prediction": risk_level,
            "confidence": round(confidence, 4)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
