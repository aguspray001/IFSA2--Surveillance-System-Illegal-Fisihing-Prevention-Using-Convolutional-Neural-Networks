import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd

MIN_MATCH_COUNT = 10
nilai_list = []
img1 = cv2.imread('/home/agus/feature_match/query.jpg',0)#queryImage
folder = sorted(glob.glob('*.jpg'))
for img2 in folder:
    img2 = cv2.imread(img2,0)#trainImage
    # print(folder) #cek urutan file yang terbaca
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
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

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    
#list program untuk menemukan gambar yang cocok dengan query image
match_img = nilai_list.index(max(nilai_list))
kondisi = np.count_nonzero(nilai_list)

if kondisi == 0 :
    print("tidak ada yang cocok")
else:
    print(match_img+1)
    match_img = str(match_img+1)+'.jpg'
    # print(type(match_img))
    cocok = cv2.imread(match_img,0)
    # print(type(cocok))
    plt.imshow(cocok,'gray') #plot gambar yang terbaik nilai presisinya
    # plt.plot(nilai_list) untuk plot grafik
    # plt.ylabel('nilai')
    plt.show()
