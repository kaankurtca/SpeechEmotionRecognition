# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:33:42 2021

@author: osman
"""

import sounddevice as sd
import soundfile as sf
import librosa
from playsound import playsound
import numpy as np

def extract_feature(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result



sampleRate = 16000
duration = 3.3
filename = "deneme3.wav"

print("start speak")

mydata = sd.rec(int(sampleRate * duration), samplerate=sampleRate,
    channels=1, blocking=True)/5
print("end")
sd.wait()
sf.write(filename, mydata, sampleRate)



playsound(filename)

feature = np.array(extract_feature(filename, mfcc=True, chroma=True, mel=True))

np.save("C:/Users/osman/PycharmProjects/SpeechProcessing/x_newtest",feature)




