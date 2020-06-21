import numpy as np
import cv2 as cv
import sys
import numpy as np
import os.path
from glob import glob

frame_count = 0
frame_count_out=0
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

helmets = []


classesFile = "obj.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolov3-obj.cfg";
modelWeights = "yolov3-obj_2400.weights";

args = ['yolov3.txt','yolov3.cfg', '','yolov3.weights']

net = cv.dnn.readNet(args[3], args[1])
net1 = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, label):

    if label == "SAFE":
        color = [0, 255, 0]
    else:
        color = [0, 0, 255]

    cv.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv.putText(img, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def doOverlap(l1, r1, l2, r2): 
     
    if(l1.x >= r2.x or l2.x >= r1.x): 
        return False
  
    if(l1.y <= r2.y or l2.y <= r1.y): 
        return False
  
    return True

def touching(x1, y1, w1, h1, x2, y2, w2, h2):
    l1 = Point(x1, y1 + h1) 
    r1 = Point(x1 + w1, y1) 
    l2 = Point(x2, y2 + h2) 
    r2 = Point(x2 + w2, y2) 

    return doOverlap(l1, r1, l2, r2)



def drawPred(image, classId, conf, left, top, right, bottom):

    global frame_count
    cv.rectangle(image, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf

    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])

    label_name,label_conf = label.split(':')
    if label_name == 'Helmet':
        #cv.rectangle(image, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        #cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        frame_count+=1

    if(frame_count> 0):
        return frame_count

def postprocess(frame, outs):

    global helmets 

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out=0
    classIds = []
    confidences = []
    boxes = []

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person=0
    helmets = boxes
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        frame_count_out = drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        
        my_class='Helmet'
        unknown_class = classes[classId]

        if my_class == unknown_class:
            count_person += 1


def predict(image_path):
    
    frame_count = 0
    frame_count_out=0
    confThreshold = 0.5
    nmsThreshold = 0.4
    inpWidth = 416
    inpHeight = 416
    global helmets
    helmets = []



    image = cv.imread(image_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    classes = None
    with open(args[0], 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    blob = cv.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    blob1 = cv.dnn.blobFromImage(image, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net1.setInput(blob)
    outs1 = net1.forward(get_output_layers(net1))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    postprocess(image, outs1)

    final = 0
    num_helmets = 0

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        is_safe = "SAFE"
        

        if class_ids[i] == 0:
            final += 1
            
            for k in helmets:
                if k[0] >= x and k[0] + k[2] <= x + w:
                    num_helmets += 1

            if num_helmets < final:
                is_safe = "UNSAFE"

            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), is_safe)

    cv.imwrite("/static/new_output.jpg", image)

    if num_helmets > final:
        return image, final, final
    return image, final, num_helmets
<<<<<<< HEAD
=======

print(predict("input.jpg"))
predict("input1.jpg")

>>>>>>> e713eccdb846c591cecc2106606f0a8bb56e1eb0
