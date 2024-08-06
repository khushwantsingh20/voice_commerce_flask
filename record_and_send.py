import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests

def record_audio(filename, duration, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wav.write(filename, fs, recording)
    print(f"Recording saved to {filename}")

def send_audio_to_api(filename, url):
    with open(filename, 'rb') as audio_file:
        files = {'audio': audio_file}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print("Text:", response.json().get('text'))
        else:
            print("Error:", response.json().get('error'))

if __name__ == '__main__':
    audio_filename = 'recorded_audio.wav'
    api_url = 'http://127.0.0.1:5000/convert'
    
    # Record audio
    duration = 5  # seconds
    record_audio(audio_filename, duration)
    
    # Send recorded audio to the Flask API
    send_audio_to_api(audio_filename, api_url)
