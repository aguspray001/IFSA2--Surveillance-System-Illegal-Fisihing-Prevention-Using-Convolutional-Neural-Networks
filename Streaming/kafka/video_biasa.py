import time
import cv2
from imutils.video import FPS

video = cv2.VideoCapture(-1)
fps = FPS().start()

if(video.isOpened()==False):
    print("unable to read camera feed")

while(True):
    success, frame = video.read()
    resize = cv2.resize(frame, (300, 300)) 
    cv2.imshow("video_biasa", resize)

    if cv2.waitKey(1)== ord('q'):
        break
    fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
#video.release()
print('sukses!')
