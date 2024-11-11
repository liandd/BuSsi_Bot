#!/usr/bin/env python3

import speech_recognition as sr
import pyglet as pt
import time
import os
from gtts import gTTS
from pydub import AudioSegment

# Globales
r = sr.Recognizer()

class ChatbotControlador:
    def __init__(self):
        self.audio_dir = os.path.join(os.path.dirname(__file__), '..','static')

    def speech_file(self):
        ogg_file = os.path.join(self.audio_dir, 'audios_ogg', 'Bussi_pregunta2.ogg')
        wav_file = os.path.join(self.audio_dir, 'audios_wav', 'Prueba2.wav')

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

controlador = ChatbotControlador()
str_de_audio = controlador.speech_file()
wav_path = controlador.generate_audio(str_de_audio)
music = pt.media.load(wav_path, streaming=False)
music.play()
time.sleep(music.duration)
print('SALIENDO.')
