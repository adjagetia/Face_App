import cv2
import numpy as np
import os
import glob
from random import shuffle
import csv


if not os.path.exists("dataset"):
    os.system("mkdir dataset")

fpath = glob.glob("videos/*/*")

fname = [os.path.basename(os.path.dirname(x)) for x in fpath]

for f in fname:
    newpath_object = "dataset/" + f
    if not os.path.exists(newpath_object):
        os.makedirs(newpath_object)

k=0
l = len(fpath)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for i in range(l):
    cap = cv2.VideoCapture(fpath[i])
    print(k)
    print("going into")
    print(fname[i])
    print(fpath[i])
    frame_number=0
    while(cap.isOpened()):
        ret,frame = cap.read()
        if not ret:
            cap.release()
            break
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame,1)
        newx,newy = frame.shape[1]/4,frame.shape[0]/4
        frame = cv2.resize(frame,(int(newx),int(newy)))
        # k = len(os.listdir("dataset/" + fname[i] + "/"))
        # cv2.imwrite("dataset/" + fname[i] + "/" + fname[i] + str(k) + '.jpg',frame)
        # print("hello")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        o=0;
        for (x,y,w,h) in faces:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
            roi_color = frame[y:y+h, x:x+w]
            o=o+1;
            roi_color = cv2.resize(roi_color,(50,50))

        if o>0:
            k = len(os.listdir("dataset/" + fname[i] + "/"))
            frame_number=frame_number+1;
            if frame_number>1000:
                break
            # print(fname[i])
            # print(fpath[i])
            cv2.imwrite("dataset/" + fname[i] + "/" + fname[i] + str(k) + '.jpg',roi_color)



image_path = glob.glob("dataset/*/*")
shuffle(image_path)
ilabel = [os.path.basename(os.path.dirname(x)) for x in image_path]
rows = zip(image_path,ilabel)
with open("dataset/face.csv", "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

print("done")

# for path in fpath:
#
#
#
# for path in xmlpath:
#     obj = untangle.parse(path)
#     image_name = obj.annotation.filename.cdata
#     image_dir = test + "JPEGImages/" + str(image_name)
#     img = cv2.imread(image_dir,0)
#     # cv2.imshow('image',img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     objects = obj.annotation.object
#     for child in objects:
#         obj_name = child.name.cdata
#         newpath_object = "output/" + obj_name
#         if not os.path.exists(newpath_object):
#             os.makedirs(newpath_object)
#         xmin = int(float(child.bndbox.xmin.cdata))
#         ymin = int(float(child.bndbox.ymin.cdata))
#         xmax = int(float(child.bndbox.xmax.cdata))
#         ymax = int(float(child.bndbox.ymax.cdata))
#         crop_img = img[ymin:ymax, xmin:xmax]
#         resized_image = cv2.resize(crop_img, (224, 224), interpolation = cv2.INTER_LANCZOS4)
#
#         i = len(os.listdir(newpath_object))
#
#         image_path = newpath_object + "/" + str(i) + ".jpg"
#         cv2.imwrite(image_path,resized_image)#,[int(cv2.IMWRITE_PXM_BINARY), 1])
#
# # pickleFileName = "Pickles/boxData_run.pickle"
# # pickleFile = open("bin", 'wb')
# # pickle.dump("output",pickleFile)
# # pickleFile.close()
