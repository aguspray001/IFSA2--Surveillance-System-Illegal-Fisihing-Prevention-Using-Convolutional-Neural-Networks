#library GUI
import os, sys
sys.path.append('..')
import math
import warnings
import requests
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import datetime
import platform
from pyodm import Node, exceptions
#Library deteksi dan generate coordinate
from osgeo import osr,gdal
import numpy as np
import tensorflow as tf
import cv2 as cv
import glob
import os
from matplotlib import pyplot as plt
#geo plotting
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
#interactive geo plot
import folium
from folium.plugins import MarkerCluster
import webbrowser
OS = platform.system()

#dimensi window utama
HEIGHT = 600
WIDTH = 800
#waktu
now = datetime.datetime.now()
seconds= now.second
##########################################################################################################################
#fungsi buka dan pilih dataset pemetaan
def OpenFile():
    global open_file
    open_file = filedialog.askopenfilenames(initialdir = "/home/agus/pemetaan", title="pilih file", filetypes = (("JPG", "*.JPG"),("all files", "*.*")))
    open_file = list(open_file) #convert to list

def OpenFile1():
    global open_file1
    open_file1 = filedialog.askopenfilename(initialdir = "/home/agus/feature_match", title="pilih file", filetypes = (("jpg", "*.jpg"),("all files", "*.*")))
    app = MainWindow(canvas_main4, path=open_file1)

#fungsi membuka hasil pemetaan (button show images)
def openResults():
    global open_results
    open_results = filedialog.askopenfilename(initialdir = "/home/agus", title="file hasil", filetypes = (("tif images", "*.tif"),("all files", "*.*")))
    # open_results = list(open_results)
    app = MainWindow(canvas_main, path=open_results)
    print(open_results)

#popup message
def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("Message")
    label=ttk.Label(popup, text=msg, font=40)
    label.pack(side="top", fill="x", pady=10)
    B1= ttk.Button(popup, text="close", command = popup.destroy)
    B1.pack()
    popup.mainloop()

# def open_map():  #plot uninteractive geographic
#     map_plot = tk.Tk()
#     map_plot.wm_title("IF-SA02 MAP")

#     fig = Figure(figsize=(7, 5), dpi=100)
#     ax = fig.add_subplot(111)
#     m = Basemap(projection='lcc', resolution=None,
#             width=4E6, height=4E6, 
#             lat_0=45, lon_0=-100,ax=ax)

#     m.bluemarble(scale=.5, alpha=.5)
#     lat_dummy = [-82.7523169603922, -82.7423169603922, -82.7123169603922, -81.7523169603922, -85.7523169603922, -84.7523169603922, -85.5523169603922]
#     lon_dummy = [41.30364938833841, 41.30364938833841, 41.30364938833841, 41.30364938833841, 41.50364938833841, 41.90364938833841, 41.70364938833841]
#     # Map (long, lat) to (x, y) for plotting
#     x, y = m(lat_list, lon_list)
#     x_dummy, y_dummy = m(lat_dummy, lon_dummy)
#     # x1, y1 = m(-81.7523169603922, 41.30364938833841)
#     ax.plot(x, y, 'or', markersize=10, alpha=.5, label = "Unreported")
#     ax.plot(x_dummy, y_dummy, 'og', markersize=10, alpha=.5, label = 'Reported')
#     ax.legend(loc = 'upper left')

#     # a tk.DrawingArea
#     canvas_map = FigureCanvasTkAgg(fig, master=map_plot)
#     canvas_map.draw()
#     canvas_map.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

#     toolbar = NavigationToolbar2Tk(canvas_map, map_plot)
#     toolbar.update()
#     canvas_map._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
#     button = tk.Button(master=map_plot, text='Quit', command=map_plot.destroy)
#     button.pack(side=BOTTOM)
#     tk.mainloop()

def interactive_map():
    import itertools
    filepath = '/home/agus/pemetaan/parseloc/interactivate_map.html'
    boulder_coords = [41.30494175603432,-81.75212109949386]
    latlon_dummy = [(41.30664938833841, -81.7523169603922),(41.30694175603432,-81.75212109949386)]
    polygon_map = folium.Map(location=boulder_coords, zoom_start = 15)
    for coord_detect, dummy in itertools.product(coor_map, latlon_dummy):
        folium.Marker(location = [coord_detect[0], coord_detect[1]], popup=coord_detect,icon=folium.Icon(color='red', icon='info-sign')).add_to(polygon_map)
        folium.Marker(location = [dummy[0], dummy[1]], popup=dummy,icon=folium.Icon(color='green', icon='info-sign')).add_to(polygon_map)
    polygon_map.save(filepath)
    webbrowser.open(filepath)

# def streaming_kafka():
#     from kafka import KafkaConsumer
#     from imutils.video import FPS
#     consumer = KafkaConsumer('videoStream', bootstrap_servers=['192.168.100.29:9092'])
#     fps = FPS().start()
#     def get_video():
#         for message in consumer:
#             frame = np.frombuffer(message.value, dtype='uint8')
#             yield cv2.imdecode(frame, cv2.IMREAD_COLOR)

#         for value in get_video():
#             value = cv2.resize(value, (300,300))
            
#             cv2.imshow('frame', value)

#             if cv2.waitKey(1)== ord('q'):
#                 break
#             fps.update()
#         fps.stop()
#         print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#         print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#         cv2.destroyAllWindows()

def help_instruction():
    bantuan = tk.Tk()
    bantuan.title("IF-SA02 - Help Instruction")
    bantuan = tk.Text(bantuan, height = 20, width=100)
    bantuan.pack()
    quote = """========Mapping Instruction========
1. Input images from your dataset through "input image" button
2. Click "start mapping" button
3. Wait until show a popup message "mapping is success!"
4. The mapping result automatically save  into your directory

========Object Detection Instruction========
1. Click "start detect" button
2. Wait until show a popup message "detecting is success!"
3. The data such as location, date/time, and 
   number of vessels automatically save into your directory
   in .txt file.

========Check the Hull Number Plate Instruction========
1. Click check plate button in main menu
2. Input query image or captured image
3. Click "start matching" button
4. The mathing precision data will show in display in percent form"""
    bantuan.insert(tk.END, quote)
    bantuan.config(state= DISABLED) #read-only
    bantuan.mainloop()
########################################################################################################################
#fungsi pemetaan    
def start_mapping():
    global kondisi
    kondisi = 'start'
    if kondisi == 'start':
        node = Node("localhost", 3000)
        try:
            # Start a task
            print("Uploading images...")
            task = node.create_task(open_file,{'dsm': True, 'orthophoto-resolution': 3})
            print(task.info())

            try:
                # This will block until the task is finished
                # or will raise an exception
                task.wait_for_completion()

                print("Task completed, downloading results...")

                # Retrieve results
                task.download_assets("results")

                print("Assets saved in ./results (%s)" % os.listdir("results"))

                # Restart task and this time compute dtm
                task.restart({'dtm': True})
                task.wait_for_completion()

                print("Task completed, downloading results...")

                task.download_assets("./results_with_dtm")

                print("Assets saved in ./results_with_dtm (%s)" % os.listdir("./results_with_dtm"))
            except exceptions.TaskFailedError as e:
                print("\n".join(task.output()))

        except exceptions.NodeConnectionError as e:
            print("Cannot connect: %s" % e)
        except exceptions.NodeResponseError as e:
            print("Error: %s" % e)


#fungsi plate recognition menggunakan feature matching
def match_gambar():
    MIN_MATCH_COUNT = 10
    nilai_list = []
    img1 = cv.imread(open_file1,0)#queryImage
    folder = sorted(glob.glob('/home/agus/feature_match/dataset/*.jpg'))
    for img2 in folder:
        img2 = cv.imread(img2,0)#trainImage
        # print(folder) #cek urutan file yang terbaca
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
        search_params = dict(checks = 50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)


        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)

            img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
            #print(mask)
            inliers = np.sum(mask)
            matched = len(mask)
            value_FM = (inliers/matched)*100
            # print('%d / %d  inliers/matched' % (inliers,matched))
            print("akurasi: %f" %(value_FM))#print akurasi matching (inliers/matched)
            # print("precision: %f" %(1-((matched-inliers)/matched)))
        else:
            # print("Not enough matches are found : %d/%d" % (len(good),MIN_MATCH_COUNT)) #good keypoint/minimal keypoint
            matchesMask = None
            value_FM = 0
            print("akurasi: %f" %(value_FM))

        nilai_list.append(value_FM)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

        img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        # plt.imshow(img3,'gray'),plt.show()
    #list program untuk menemukan gambar yang cocok dengan query image    
    match_img = nilai_list.index(max(nilai_list))
    max_value = max(nilai_list)
    nilai_max = str(max_value)+'%'
    nilai_ilegal = str(0)+'%'
    kondisi = np.count_nonzero(nilai_list)

    if kondisi == 0 or max_value < 75:
        print("tidak ada kecocokan dengan dataset")
        matching_precision.delete(0,END)
        matching_precision.insert(0,nilai_ilegal)
        app = MainWindow(canvas_main3, path='noimage.png')
        popupmsg("Illegal Fishing!")

    else:
        print(match_img+1)
        match_img = '/home/agus/feature_match/dataset/'+str(match_img+1)+'.jpg'
        # print(match_img)
        cocok = cv.imread(match_img,0)
        # print(cocok)
        app = MainWindow(canvas_main3, path=match_img)
        matching_precision.insert(0,nilai_max)
        popupmsg("This Is Not Illegal Fishing")
        # plt.imshow(cocok,'gray'),plt.show()

#fungsi deteksi dan genereate koordinat
def detectxcoor():
    # Read the graph.
    global lat_list
    global lon_list
    global coor_map
    with tf.gfile.FastGFile('/home/agus/ComputerVision/kapal_3497_siap/mobilenet_data/model_mobilenet.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.

        # for file in glob.glob("*.jpg"):
        #img = cv.imread(file)
        img = cv.imread('/home/agus/pemetaan/results/odm_orthophoto/orthophoto_fake.jpg') #input hasil mapping .png atau .tif
        #print img
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (400, 400))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        ds = gdal.Open('odm_orthophoto.tif') #hasil mapping format .tif
        old_cs= osr.SpatialReference()
        old_cs.ImportFromWkt(ds.GetProjectionRef())

        # create the new coordinate system
        wgs84_wkt = """
        GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-81],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32617"]]"""
        new_cs = osr.SpatialReference()
        new_cs .ImportFromWkt(wgs84_wkt)

        # create a transform object to convert between coordinate systems
        transform = osr.CoordinateTransformation(old_cs,new_cs) 

        # Visualize detected bounding boxes.
        coor_list = []
        coor_map = []
        lat_list = []
        lon_list = []
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.9:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=20)
                
                value = ["ship", score, x, y, right, bottom]
                mx = int((right+x)/2)
                my = int((y+bottom)/2)

                gt = ds.GetGeoTransform()
                x_min = gt[0]
                x_size = gt[1]
                y_min = gt[3]
                y_size = gt[5]

                # mx, my = 0000, 0000  #coord in map units, as in question
                px = mx * x_size + x_min #x pixel
                py = my * y_size + y_min #y pixel
                # print(mx, my)
                # print (px, py)
                # Center coordinates 
                center_coordinates = (mx, my) 
                
                # Radius of circle 
                radius = 50
                
                # Red color in BGR 
                color = (0, 0, 255) 
                
                # Line thickness of -1 px 
                thickness = -1
                
                # Using cv2.circle() method 
                # Draw a circle of red color of thickness -1 px 
                img = cv.circle(img, center_coordinates, radius, color, thickness) 

                # get CRS from dataset 
                crs = osr.SpatialReference()
                crs.ImportFromWkt(ds.GetProjectionRef())
                # create lat/long crs with WGS84 datum
                crsGeo = osr.SpatialReference()
                crsGeo.ImportFromEPSG(4326) # 4326 is the EPSG id of lat/long crs 
                t = osr.CoordinateTransformation(crs, crsGeo)
                (lat, long, z) = t.TransformPoint(px, py)
                index = int(i+1)
                # coor_kapal = index,lat,long
                coor_kapal = lat, long
                coor_label = index, lat, long
                coor_date = index,center_coordinates,lat,long,str(now)
                label_coor =str(coor_label)
                cv.putText(img,label_coor, (int(x), int(y-10)), cv.FONT_HERSHEY_SIMPLEX, 5, (36,255,12), 10)
                # print(lat, long, z)
                coor_list.append(coor_date)
                coor_map.append(coor_kapal)
                # print(coor_list)
                lat_list.append(lat)
                lon_list.append(long)
        text_file = open("Output.txt", "w")
        text_file.write("format:(index;latitude;longitude;date&time)= %s" %coor_list)
        text_file.close() 
        cv.imwrite("hasil_deteksi.jpg", img)
        lokasi_hasil = "/home/agus/pemetaan/parseloc/hasil_deteksi.jpg"
        kondisi_file = os.path.exists('/home/agus/pemetaan/parseloc/hasil_deteksi.jpg')
        
        if kondisi_file == True:            
            app = MainWindow(canvas_main, path=lokasi_hasil) # Displaying the image in GUI
            jumlah_kapal.insert(0,str(index))
            time_detection.insert(0,str(now))
            popupmsg("Detection is success!")  
            #sukses

#download images from storage
def download_image():
    import pyrebase
    import glob, os, sys
    import firebase_admin
    from firebase_admin import credentials
    from datetime import datetime
    cred = credentials.Certificate("/home/agus/gambar/roboport-pens-firebase-adminsdk-yzhb3-7c7b98c11d.json")
    firebase_admin.initialize_app(cred)

    config = {
        "apiKey": "AIzaSyDdjc3ESqNTNF0e1rIrLuoPc7iz0IDg2ts",
        "authDomain": "roboport-pens.firebaseapp.com",
        "databaseURL": "https://roboport-pens.firebaseio.com",
        "projectId": "roboport-pens",
        "storageBucket": "roboport-pens.appspot.com",
        "messagingSenderId": "737779930108",
        "appId": "1:737779930108:web:c65ec9f1e34539f4f99879",
        "measurementId": "G-NFB28PGN2V",
        "serviceAccount": "/home/agus/gambar/roboport-pens-firebase-adminsdk-yzhb3-7c7b98c11d.json"
    }

    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()

    # folder = sorted(glob.glob('/home/agus/gambar/*.JPG'))
    folder = storage.list_files()
    number_of_images = []
    start = datetime.now()
    for img in folder:
        number_of_images.append(img)
        n_images = len(number_of_images)
        print(n_images)
        path_on_cloud = "/images/"+str(n_images)+".JPG"
        path_on_local = "/home/agus/gambar/"+str(n_images)+".JPG"
        # print(path_on_cloud, path_on_local)
        try:
            storage.child(path_on_cloud).download(path_on_local)
            print(datetime.now()-start)
        except:
            print("download failed")
    print("downloaded: "+str(n_images)+" files.....") 
########################################################################################################################
#fungsi membuka window baru
def openWindow():
    global canvas_main3 #train
    global canvas_main4 #query
    global matching_precision
    global gambarx
    top = Toplevel()
    top.title('IF-SA02 - Plate Recognition')
    p2 = PhotoImage(file = 'uav.png') #cuma buat manggil aja
    top.iconphoto(False, p2)
    top.geometry('800x600')
    #canvas top
    canvas_main2 = Canvas(top, height=HEIGHT, width=WIDTH, bg='#00a8cc')
    canvas_main2.pack(expand= YES, fill=BOTH)
    #init image
    gambar2 = Image.open('background.jpg')
    gambar2 = gambar.resize((1000,1000), Image.ANTIALIAS)
    background_image2 = ImageTk.PhotoImage(gambar2)
    background_label2 = Label(canvas_main2, image=background_image)
    background_label2.place(relwidth=1, relheight=1)
    #
    menu = Menu(top)
    top.config(menu=menu)
    filemenu= Menu(menu)
    helpmenu= Menu(menu)
    modemenu= Menu(menu)
    menu.add_cascade(label="File", menu=filemenu)
    filemenu.add_command(label = "Open File", command= OpenFile1)
    filemenu.add_separator()
    filemenu.add_command(label='Exit', command= top.destroy)
    # menu.add_cascade(label="Mode", menu=modemenu, command= help_instruction)
    # modemenu.add_command(label='Plate Recognition', command= openWindow)
    menu.add_cascade(label="Help?", menu=helpmenu, command= help_instruction)
    helpmenu.add_command(label='Help', command= help_instruction)
    #
    labelText=StringVar()
    labelText.set("Query Image")
    labelDir=Label(top, textvariable=labelText, height=4, background='#4cd3c2')
    labelDir.place(relx=0.5, rely=0.41, relheight=0.03, relwidth=0.2, anchor = 'n')
    #
    main_frame2 = tk.Frame(top, bg='#00a8cc', bd=8)
    main_frame2.place(relx=0.5, rely=0.45, relwidth=0.4, relheight=0.3, anchor='n')
    #
    labelText=StringVar()
    labelText.set("Output Image")
    labelDir=Label(top, textvariable=labelText, height=4, background='#4cd3c2')
    labelDir.place(relx=0.5, rely=0.76, relheight=0.03, relwidth=0.2, anchor = 'n')
    #
    main_frame3 = tk.Frame(top, bg='#00a8cc', bd=8)
    main_frame3.place(relx=0.5, rely=0.1, relwidth=0.4, relheight=0.3, anchor='n')
    #
    canvas_main3 = Canvas(main_frame2, height=HEIGHT, width=WIDTH, bg='#00a8cc')
    canvas_main3.pack(fill=BOTH,expand=YES)
    #
    canvas_main4 = Canvas(main_frame3, height=HEIGHT, width=WIDTH, bg='#00a8cc')
    canvas_main4.pack(fill=BOTH,expand=YES)
    #
    gambarx = Image.open('noimage.png')
    gambarx = gambarx.resize((150,150), Image.ANTIALIAS)
    main_imagex = ImageTk.PhotoImage(gambarx)
    no_image1 = Label(canvas_main3, image=main_imagex, bd=0)
    no_image1.place(relwidth=1, relheight=1)
    no_image2 = Label(canvas_main4, image=main_imagex, bd=0)
    no_image2.place(relwidth=1, relheight=1)
    #matching precision display
    matching_precision= tk.Entry(top)
    matching_precision.place(relx =0.75, rely=0.3, relheight=0.05, relwidth=0.2)
    #
    labelText=StringVar()
    labelText.set("Matching Precision(%)")
    labelDir=Label(top, textvariable=labelText, height=4, background='#5fdde5')
    labelDir.place(relx =0.75, rely=0.35, relheight=0.03, relwidth=0.2)
    #
    button_start2 = tk.Button(top, text="Start Match", font=40, activebackground='#639a67', command=match_gambar)
    button_start2.place(relx =0.37, rely=0.85, relheight=0.05, relwidth=0.25)
    #
    # button_start3 = tk.Button(top, text="Streaming", font=40, activebackground='#639a67', command=match_video)
    # button_start3.place(relx =0.5, rely=0.85, relheight=0.05, relwidth=0.25)
    # 
    # labelText=StringVar()
    # labelText.set("Select Mode:")
    # labelDir=Label(top, textvariable=labelText, height=4, background='#5fdde5', )
    # labelDir.place(relx =0.1, rely=0.85, relheight=0.05, relwidth=0.15)
    # button_exit2 = tk.Button(top, text="Exit", font=40, activebackground='#d63447', command= top.destroy)
    # button_exit2.place(relx =0.37, rely=0.9, relheight=0.05, relwidth=0.25)

########################################################################################################################
#fungsi scrollbar pada canvas
class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

class CanvasImage:
    """ Display and zoom image """
    def __init__(self, placeholder, path):
        """ Initialize the ImageFrame """
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.path = path  # path to the image, should be public for outer classes
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe,bg='#dee3e2', highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>',     self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.__wheel)  # zoom for Linux, wheel scroll up
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning for big image
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)  # open image, but down't load it into RAM
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
           self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        (w, h), m, j = self.__pyramid[-1].size, 512, 0
        n = math.ceil(math.log(min(w, h) / m, self.__reduction)) + 1  # image pyramid length
        while w > m and h > m:  # top pyramid image is around 512 pixels in size
            j += 1
            print('\rCreating image pyramid: {j} from {n}'.format(j=j, n=n), end='')
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        print('\r' + (40 * ' ') + '\r', end='')  # hide printed string
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 0, math.ceil(self.imheight / self.__band_width)
        while i < self.imheight:
            j += 1
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k)+1), self.__filter), (0, int(i * k)))
            i += band
        print('\r' + (40 * ' ') + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if  box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0]  = box_img_int[0]
            box_scroll[2]  = box_img_int[2]
        # Vertical part of the image is in the visible area
        if  box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1]  = box_img_int[1]
            box_scroll[3]  = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image, which does not fit in RAM
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                                    (int(x1 / self.__scale), int(y1 / self.__scale),
                                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        if OS == 'Darwin':
            if event.delta<0:  # scroll down, zoom out, smaller
                if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
                self.imscale /= self.__delta
                scale        /= self.__delta
            if event.delta>0:  # scroll up, zoom in, bigger
                i = float(min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1)
                if i < self.imscale: return  # 1 pixel is bigger than the visible area
                self.imscale *= self.__delta
                scale        *= self.__delta
        else:
            # Respond to Linux (event.num) or Windows (event.delta) wheel event
            if event.num == 5 or event.delta == -120:  # scroll down, zoom out, smaller
                if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
                self.imscale /= self.__delta
                scale        /= self.__delta
            if event.num == 4 or event.delta == 120:  # scroll up, zoom in, bigger
                i = float(min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1)
                if i < self.imscale: return  # 1 pixel is bigger than the visible area
                self.imscale *= self.__delta
                scale        *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right, keys 'd' or 'Right'
                self.__scroll_x('scroll',  1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left, keys 'a' or 'Left'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up, keys 'w' or 'Up'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down, keys 's' or 'Down'
                self.__scroll_y('scroll',  1, 'unit', event=event)

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        if self.__huge:  # image is huge and not totally in RAM
            band = bbox[3] - bbox[1]  # width of the tile band
            self.__tile[1][3] = band  # set the tile height
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3  # set offset of the band
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # image is totally in RAM
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()
        self.__imframe.destroy()

class MainWindow(ttk.Frame):
    """ Main window class """
    def __init__(self, mainframe, path):
        """ Initialize the main Frame """
        ttk.Frame.__init__(self, master=mainframe)
        # self.master.title('Advanced Zoom v3.0')
        # self.master.geometry('800x600')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)
        canvas = CanvasImage(self.master, path)  # create widget
        canvas.grid(row=0, column=0)  # show widget
# filename = 'odm_orthophoto.tif'  # place path to your image here

######################################################################################################################## 
root = tk.Tk()
root.title("IF-SA02 - Mapping and Detection")
p1 = PhotoImage(file = 'uav.png') #cuma buat manggil aja
root.iconphoto(False, p1)
#
menu = Menu(root)
root.config(menu=menu)
file_menu= Menu(menu)
helpmenu= Menu(menu)
modemenu= Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label = "Input File..", command= OpenFile)
file_menu.add_separator()
file_menu.add_command(label='Open Mapping', command= interactive_map)
file_menu.add_separator()
file_menu.add_command(label='Exit', command= root.destroy)
menu.add_cascade(label="Mode", menu=modemenu, command= help_instruction)
modemenu.add_command(label='Plate Recognition', command= openWindow)
menu.add_cascade(label="Help?", menu=helpmenu, command= help_instruction)
helpmenu.add_command(label='Help', command= help_instruction)
#
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack(expand= YES, fill=BOTH)
#init image
gambar = Image.open('background.jpg')
gambar = gambar.resize((1800,1800), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(gambar)
background_label = Label(canvas, image=background_image)
background_label.place(relwidth=1, relheight=1)

#frame utama
main_frame = tk.Frame(root, bg='#00a8cc', bd=6)
main_frame.place(relx=0.4, rely=0.05, relwidth=0.75, relheight=0.6, anchor='n')
#mengisi frame dengan canvas yang berisi gambar noimage
canvas_main = Canvas(main_frame, height=300, width=300, bg='#00a8cc')
canvas_main.pack(expand= YES, fill=BOTH)
#inisialisasi image
gambar = Image.open('noimage.png')
gambar = gambar.resize((300,300), Image.ANTIALIAS)
main_image = ImageTk.PhotoImage(gambar)
#menampilkan gambar dalam canvas
# canvas_main.create_image(150,30, image = main_image, anchor= 'nw')
no_image = Label(canvas_main, image=main_image,bd=0)
no_image.place(relwidth=1, relheight=1)


#mengisi frame dengan fitur entry dan button
#atas
# button_input = tk.Button(root, text="Input Image", font=40, command= OpenFile)
# button_input.place(relx = 0.055,rely=0.7, relwidth=0.2, relheight=0.06)
button_start = tk.Button(root, text="Start Mapping", font=40, command=start_mapping, activebackground='#639a67')
button_start.place(relx =0.2,rely=0.7, relwidth=0.2, relheight=0.06)
# button_show = tk.Button(root, text="Show Image", font=40, command=openResults)
# button_show = tk.Button(root, text="Help?", font=38, command=help_instruction)
# button_show.place(relx=0.55,rely=0.7, relwidth=0.2, relheight=0.06)
#bawah
# button_featureMatching = tk.Button(root, text="Plate Recognition", font=40, command=openWindow)
# button_featureMatching.place(relx = 0.055,rely=0.8, relwidth=0.2, relheight=0.06)
button_detection = tk.Button(root, text="Detection", font=40, command=detectxcoor, activebackground='#639a67')
button_detection.place(relx = 0.4,rely=0.7, relwidth=0.2, relheight=0.06)
# button_exit = tk.Button(root, text="Exit", font=40, command=root.destroy, activebackground='#d63447')
# button_exit.place(relx = 0.55,rely=0.8, relwidth=0.2, relheight=0.06)

#samping
# data_frame = tk.Frame(root, bg='#00a8cc', bd=5) #membuat frame
# data_frame.place(relx=0.99, rely=0.35, relwidth=0.2, relheight=0.6, anchor='e')
#label teks
labelText=StringVar()
labelText.set("Jumlah Kapal")
labelDir=Label(root, textvariable=labelText, height=4, background='#5fdde5')
labelDir.place(relx =0.8, rely=0.15, relheight=0.05, relwidth=0.15)
#data jumlah kapal
jumlah_kapal = tk.Entry(root)
jumlah_kapal.place(relx =0.8, rely=0.1, relheight=0.05, relwidth=0.15)
#label teks
labelText=StringVar()
labelText.set("Date and Time")
labelDir=Label(root, textvariable=labelText, height=4, background='#5fdde5')
labelDir.place(relx =0.8, rely=0.35, relheight=0.05, relwidth=0.15)
#date and time
time_detection = tk.Entry(root)
time_detection.place(relx =0.8, rely=0.3, relheight=0.05, relwidth=0.15)
#button help/ reset entry
button_help = tk.Button(root, text="Help?", font=38, command=help_instruction)
button_help.place(relx =0.8, rely=0.6, relheight=0.05, relwidth=0.15)
#button download image from firebase
button_download = tk.Button(root, text="Load images", font=38, command=download_image)
button_download.place(relx =0.8, rely=0.5, relheight=0.05, relwidth=0.15)
root.mainloop()
