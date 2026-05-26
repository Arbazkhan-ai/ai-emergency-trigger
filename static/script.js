let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const micBtn = document.getElementById('micBtn');
const statusText = document.getElementById('statusText');
const recordingIndicator = document.getElementById('recordingIndicator');
const resultsSection = document.getElementById('resultsSection');
const loader = document.getElementById('loader');

const transcriptionText = document.getElementById('transcriptionText');
const riskLevel = document.getElementById('riskLevel');
const confidenceBadge = document.getElementById('confidenceBadge');
const emergencyAlert = document.getElementById('emergencyAlert');

micBtn.addEventListener('click', toggleRecording);

async function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = processAudio;

        mediaRecorder.start();
        isRecording = true;
        
        // UI Updates
        micBtn.classList.add('recording');
        statusText.classList.add('hidden');
        recordingIndicator.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
    } catch (err) {
        console.error("Error accessing microphone:", err);
        alert("Microphone access is required to use this feature.");
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        
        // UI Updates
        micBtn.classList.remove('recording');
        recordingIndicator.classList.add('hidden');
        loader.classList.remove('hidden');
        statusText.classList.remove('hidden');
        statusText.innerText = "Processing...";
    }
}

async function processAudio() {
    // Determine mime type, typically audio/webm on Chrome, audio/mp4 on Safari
    const mimeType = mediaRecorder.mimeType || 'audio/webm';
    const audioBlob = new Blob(audioChunks, { type: mimeType });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        displayResults(data);
    } catch (err) {
        console.error("Error sending audio to server:", err);
        alert("Failed to process audio. Please try again.");
        statusText.innerText = "Click to Start Recording";
        loader.classList.add('hidden');
    }
}

function displayResults(data) {
    loader.classList.add('hidden');
    statusText.innerText = "Click to Start Recording";
    
    // Update data
    transcriptionText.innerText = data.text;
    riskLevel.innerText = data.prediction;
    confidenceBadge.innerText = `${(data.confidence * 100).toFixed(1)}% Confidence`;
    
    // Update risk classes
    riskLevel.className = 'risk-level'; // Reset classes
    if (data.prediction.includes("HIGH")) {
        riskLevel.classList.add("HIGH");
        emergencyAlert.classList.remove("hidden");
    } else if (data.prediction.includes("MEDIUM")) {
        riskLevel.classList.add("MEDIUM");
        emergencyAlert.classList.add("hidden");
    } else {
        riskLevel.classList.add("LOW");
        emergencyAlert.classList.add("hidden");
    }
    
    resultsSection.classList.remove('hidden');
}
