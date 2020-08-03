import numpy as np
import tensorflow.compat.v1 as tf
import cv2 as cv
import glob
import os
from imutils.video import FPS, VideoStream
from time import sleep
import imutils


os.chdir("/home/agus/ComputerVision/kapal_3497_siap/mobilenet_data")
# Read the graph.
with tf.gfile.GFile('/home/agus/ComputerVision/kapal_3497_siap/frozen_mobilenet/mobilenet.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    video = cv.VideoCapture("/home/agus/ComputerVision/kapal_3497_siap/sample_uji/55m.mp4")
    # video = VideoStream(src=-1).start()
    sleep(2)
    fps = FPS().start()
    while(True):
        ret, frame = video.read() 
        # fps = video.get(cv.CAP_PROP_FPS)
        # print(fps)
        # img = cv.imread(frame)
    #img = cv.imread('/home/agus/models/research/object_detection/IF-SA02/Images/kapal1500/0001.jpg')
    #print img
    # if ret:
        img = np.array(frame)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (400, 400))
        #inp = cv.cvtColor(inp, cv.COLOR_BGR2GRAY)
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        # print(out)
    # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        deteksi = []
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            print(bbox)
            value_struct = {}
            if score > 0.1:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (255, 0, 0), thickness=8)
                cv.putText(img, str(round(score, 8))+"  kapal", (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1,(255, 255,0), 4)
                value_struct['ship'] = [score, x, y, right, bottom]
                print(value_struct)
                
                # hasil = str(value_struct)
                # hasil = hasil.replace("[","")
                # hasil = hasil.replace("]","")
                # hasil = hasil.replace("{","")
                # hasil = hasil.replace("}","")
                # hasil = hasil.replace("'","")
                # hasil = hasil.replace(",","")
                # hasil = hasil.replace(":","")

                # deteksi.append(hasil)
                # f = open(( file.rsplit( ".", 1 )[ 0 ] ) + ".txt", "w")
                # for listitem in deteksi:
                #     f.write("%s \n" %listitem)
                # f.close()
        #img = imutils.resize(img, width=1366, height=768)
        img = imutils.resize(img, width=720, height=640)
        cv.imshow('Streaming-Tensorflow', img)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv.destroyAllWindows()
# video.stop()
