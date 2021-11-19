import cv2
import numpy as np
import os

#fotoğrafları ekleme
path= 'fotograflar_orijinal'
orb = cv2.ORB_create(nfeatures=500)
images = []
classNames= []
myList = os.listdir(path)
print(myList)
print('toplam class sayisi',len(myList))

for i in myList:
    imgCur = cv2.imread(f'{path}/{i}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(i)[0])
print(classNames )

def descripter(images):
    descripter_list=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        descripter_list.append(des)
    return descripter_list

def findID(img, descripter_list, threshold=15):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1

    try:
        for des in descripter_list:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 *n.distance:
                    good.append([m])
            matchList.append(len(good))
        #print(matchList)
    except:
        pass
    if len(matchList)!=0:
        if max(matchList) > threshold:
            finalVal = matchList.index(max(matchList))
    return finalVal

descripter_list = descripter(images)
print(len(descripter_list))

cap = cv2.VideoCapture(0)

while True:
    success, img2 = cap.read()

    imgOriginal = img2.copy
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    id = findID(img2,descripter_list)
    if id != -1:
        cv2.putText(img2,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.imshow('img2',img2)
    cv2.waitKey(1)





