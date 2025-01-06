from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import librosa
import numpy as np
import torch

app = FastAPI()

# Load model and processor
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-arabic-model")
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-arabic-processor")

# Define API endpoint
@app.post("/transcribe/")
async def transcribe(audio_file: UploadFile = File(...)):
    # Load audio
    audio, sampling_rate = torchaudio.load(audio_file.file)
    
    # Resample to 16kHz
    if sampling_rate != 16000:
        audio = librosa.resample(np.asarray(audio[0].numpy()), orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    else:
        audio = audio[0].numpy()

    # Prepare input
    input_values = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    # Decode prediction
    transcription = processor.batch_decode(predicted_ids)[0]
    return {"transcription": transcription}
