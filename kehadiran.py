import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#mengambil data peserta dari folder yang sudah di sediakan
path = 'Peserta'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#encoding foto dari folder
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#Output kedalam excel
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#output saat encoding telah selesai
encodeListKnow = findEncodings(images)
print('Encoding Selesai')

#port webcam yang digunakan
cap = cv2.VideoCapture(0)

#Mencari wajah yang cocok dari value enkoding dan membuat bounding box pada program
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#membandingkan gambar yang sedang diambil dengan hasil enkoding
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

#jika nilai pada wajah secara realtime mendekati dengan nilai wajah pada enkoding maka akan match dan akan membuat bounding box
        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = 'Tidak Terdaftar'
        print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 6)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('Webcam',img)

    #memberhentikan program menggunakan huruf "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


