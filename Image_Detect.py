import cv2

img = cv2.imread('Test_Image.jpeg')
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classId, confs, bbox = net.detect(img,confThreshold=0.5)
print(classId,bbox)

for classIds, confidence, box in zip(classId.flatten(),confs.flatten(),bbox):
    cv2.rectangle(img,box,color=(0,255,0),thickness=3)
    cv2.putText(img,classNames[classIds-1].upper(),(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

cv2.imshow("Output", img)
cv2.waitKey(0)