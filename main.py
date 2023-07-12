from ultralytics import YOLO
import cv2
import math
import cvzone

cap = cv2.VideoCapture("Image.mp4")

model = YOLO("best.pt")

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
              'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan',
              'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
myColor =(0,0,255)
while True:
    success, img =cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1 ,y1 , x2, y2 = box.xyxy[0]
            x1, y1 , x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2-x1 , y2-y1
            # cvzone.cornerRect(img, (x1,y1,w,h))
            cv2.rectangle(img,(x1,y1),(x2,y2),myColor,3)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class NAme
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "Hardhat" or currentClass == "Safety Vest" or currentClass == "Mask":
                myColor = (0,255,0)
            else:
                myColor =(0,0,255)

            cvzone.putTextRect(img, f"{classNames[cls]} {conf}",
                               (max(0,x1), max(35,y1)), scale = 1,thickness =1,colorB=myColor,
                               colorT=(255,255,255),colorR=myColor,offset =5)

    cv2.imshow("Images",img)
    cv2.waitKey(1)
