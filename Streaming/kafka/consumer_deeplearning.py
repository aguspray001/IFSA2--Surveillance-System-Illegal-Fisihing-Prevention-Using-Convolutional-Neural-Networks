from kafka import KafkaConsumer
from imutils.video import FPS
import numpy as np
import cv2
import tensorflow_model as tensor_detect

consumer = KafkaConsumer('videoStream', bootstrap_servers=['192.168.100.29:9092'])
fps = FPS().start()
def get_video():
    for message in consumer:
        frame = np.frombuffer(message.value, dtype='uint8')
        yield cv2.imdecode(frame, cv2.IMREAD_COLOR)

for value in get_video():
    hasil_deteksi = tensor_detect.tensorflowDetek(value)

    cv2.imshow('frame', hasil_deteksi)

    if cv2.waitKey(1)== ord('q'):
        break
    fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
