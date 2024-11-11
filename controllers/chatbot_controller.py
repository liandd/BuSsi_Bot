#!/usr/bin/env python3

import speech_recognition as sr
import pyglet as pt
import pyaudio
import time
import sys
import os

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gtts import gTTS
from pydub import AudioSegment
from pathlib import Path
from models import chatbot_model as bussi
from utils import recorder as rd

# Globales
chatbot = bussi.Chatbot() 
r = sr.Recognizer()

class ChatbotControlador:
    def __init__(self):
        self.audio_dir = os.path.join(os.path.dirname(__file__), '..','static')

    def speech_file(self):
        ogg_file = os.path.join(self.audio_dir, 'audios_ogg', 'Hola_amigo.ogg')
        wav_file = os.path.join(self.audio_dir, 'audios_wav', 'Prueba3.wav')

        sound = AudioSegment.from_ogg(ogg_file)
        sound.export(wav_file, format='wav')

        audio_wav_file = sr.AudioFile(wav_file)
        with audio_wav_file as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)

        return r.recognize_google(audio, language='es-ES')

    def generate_audio(self, text):
        mp3_path = os.path.join(self.audio_dir, 'audios_mp3', 'audio_test1.mp3')
        wav_path = os.path.join(self.audio_dir, 'audios_wav', 'audio_test1.wav')

        tts = gTTS(text=text, lang='es')
        tts.save(mp3_path)

        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format='wav')

        return wav_path

    def start_chat(self):
        initial_text = self.speech_file()
        print(initial_text+'\n')
        self.play_audio(initial_text)
        #response = chatbot.get_response(str_de_audio)
        
        chat_generation = chatbot.chat(initial_text)
        str_response = next(chat_generation)
        print(str_response)
        self.play_audio(str_response)
        initial_text = ""

        while True:
            if initial_text == 'salir':
                exit_response = chat_generation.send(initial_text)
                print(exit_response)
                self.play_audio(exit_response)
                self.play_audio('CERRANDO CHAT!')
                break
            else:
                str_response = chat_generation.send(initial_text) if initial_text else None
                if str_response:
                    print(str_response)
                    self.play_audio(str_response)
                    continue
                    
                #initial_text = input().lower()
                initial_text = self.record_linux()

    def play_audio(self, response):
        if response:
            wav_path = self.generate_audio(response)
            music = pt.media.load(wav_path, streaming=False)
            music.play()
            time.sleep(music.duration)

    def record_windows(self):
        mic = sr.Microphone()
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        
        text = self.speech_file()
        print(text+'\n')
        return text

    def record_linux(self):
        rd.record_close()
        wav_file = os.path.join(self.audio_dir, 'audios_wav', 'from_mic.wav')

        wav_file = sr.AudioFile(wav_file)
        with wav_file as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)

        return r.recognize_google(audio, language='es-ES')
        #self.play_audio(text) 



os.system('clear')
time.sleep(1)
controlador = ChatbotControlador()
#controlador.record_windows()
controlador.start_chat()
#str_de_audio = controlador.speech_file()

#chatbot.load_model()
#padded_seq = chatbot.preprocess_input(str_de_audio)
#lem_tokens = chatbot.lem_normalize(str_de_audio)
#response = chatbot.get_response(str_de_audio)

#wav_path = controlador.generate_audio(response)
#music = pt.media.load(wav_path, streaming=False)
##print(response+'\n')
#music.play()
#time.sleep(music.duration)
print('SALIENDO.')
