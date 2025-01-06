import requests

url = "http://127.0.0.1:8000/transcribe/"
files = {'audio_file': open('C:/common_voice_ar_19238590.wav', 'rb')}
response = requests.post(url, files=files)

print(response.json())
