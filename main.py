import cv2

#img = cv2.imread('lena.png')
thres = 0.5
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classfile = 'coco.names'

with open(classfile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
print(classnames)

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightspath,configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)
    if len(classIds) != 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(255, 0, 0), thickness=2)
            cv2.putText(img, classnames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, str(round(conf*100,2)), (box[0] + 150, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('output', img)
    cv2.waitKey(1)