#!/usr/bin/env python3
import subprocess
import time

source_name = 'alsa_input.pci-0000_00_1b.0.analog-stereo'

output_file = '/home/qw4qe/Desktop/ian/Ucp/Optativa1_PLN/PROYECTO_BOT/BuSsi_Bot/static/audios_wav/from_mic.wav'

def act_mic():
    subprocess.run(['pactl','set-source-mute',source_name,'0'])

def close_mic():
    subprocess.run(['pactl','set-source-mute',source_name,'1'])

def increase_mic_volume():
    subprocess.run(['pactl', 'set-source-volume', source_name, '350%'])

def record():
    increase_mic_volume()
    subprocess.Popen(['arecord','-D','pulse','-f','cd',output_file])

def record_close():
    print('PRENDIENDO MICROFONO.')
    act_mic()

    print('GRABANDO..')
    record()
    time.sleep(7)
    print('TERMINANDO GRABACION..')
    subprocess.run(['pkill','arecord'])
    close_mic()

    print('CERRANDO')

if __name__ == "__main__":
    record_close()

