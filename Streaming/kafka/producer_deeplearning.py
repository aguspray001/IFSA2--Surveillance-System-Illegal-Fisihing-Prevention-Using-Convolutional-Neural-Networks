import time
from kafka import KafkaProducer
import tensorflow_model as fungsi_deteksi
import cv2
producer = KafkaProducer(bootstrap_servers=['192.168.100.29:9092'])

# video = cv2.VideoCapture('/home/agus/ComputerVision/video_test.mp4')
video = cv2.VideoCapture(1)
count =0 
start_time = time.time()
if(video.isOpened()==False):
    print("unable to read camera feed")

while(True):
    success, frame = video.read()
    #frame = cv2.resize(frame, (300,300))
    hasil_deteksi=fungsi_deteksi.tensorflowDetek(frame)
    ret, img = cv2.imencode('.jpg', hasil_deteksi)
    producer.send('videoStream', img.tobytes())
    #sleep(0.5)
    print("--- %s seconds ---" % (time.time() - start_time))
video.release()
print('sukses!')
