import face_recognition
import cv2
import numpy as np
import json
import os
import wave
import pyaudio
import matplotlib.pyplot as plt
import subprocess

# SetUpPyAudio(audio_file):
chunk = 1024
p = pyaudio.PyAudio()

    
def ConvertToWav(file_name,file):

    # ffmpeg\ffmpeg-20200101-7b58702-win64-static\bin
    test = subprocess.Popen(["ffmpeg\\ffmpeg-20200101-7b58702-win64-static\\bin\\ffmpeg.exe","-i",file_name, 'audio/'+file+'.wav', ], stdout=subprocess.PIPE)
    cv2.waitKey(1000)
    #command =f'ffmpeg\\ffmpeg-20200101-7b58702-win64-static\\bin\\ffmpeg.exe -i '+file_name+' audio/'+file+'.wav'
    #print(command)
    #os.system(command)
p = pyaudio.PyAudio()
path = f"dataset\dfdc_train_part_49"
req = []
with open(path+"\metadata.json") as json_file:
    data = json.load(json_file)
    #print(data)
    for x in data:
        if(data[x]["label"]=="REAL"):
            req.append(x)
    
i = 0

#req=["aamjfukxwp.mp4"]
frames = 0
current_frame = 0
for file in req:
    current_file = path+"\\"+file
    ConvertToWav(current_file,file)
    cap = cv2.VideoCapture(current_file)
    audio_file = 'audio/'+file+'.wav'
    wf = wave.open(audio_file, 'rb')
    
    while(cap.isOpened()):
        data = wf.readframes(chunk)
        
        if(frames==current_frame):
            frames+=60
            ret,img=cap.read()
            
            scale_percent=50
            try:
                i+=1
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)    
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                cv2.imshow("image",resized)
                left_channel = data[0]
                volume = np.linalg.norm(left_channel)
                print(volume)
            except:
                break
        current_frame+=1
        
        
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    """
    with audioread.audio_open(current_file) as f:
        print(f.channels, f.samplerate, f.duration)
        for buf in f:
            print(buf)
    """    
