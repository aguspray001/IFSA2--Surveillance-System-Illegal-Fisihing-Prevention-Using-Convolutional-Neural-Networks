import time
from kafka import KafkaProducer

import cv2
producer = KafkaProducer(bootstrap_servers=['10.252.133.3:9092'])

video = cv2.VideoCapture(1)
count =0 
start_time = time.time()
if(video.isOpened()==False):
    print("unable to read camera feed")

while(True):
    success, frame = video.read()
    resize = cv2.resize(frame, (300, 300)) 
    ret, img = cv2.imencode('.jpg', resize)
    producer.send('videoStream', img.tobytes())
    #sleep(0.5)
    print("--- %s seconds ---" % (time.time() - start_time))
video.release()
print('sukses!')
