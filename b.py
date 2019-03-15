from gtts import gTTS
from playsound import playsound

tts = gTTS(text="Hello crazy programmer", lang='en')
tts.save("audio.mp3")

playsound('audio.mp3')
