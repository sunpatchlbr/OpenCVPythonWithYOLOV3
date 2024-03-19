import cv2 as cv
import numpy as np
import matplotlib as plt
import argparse

Width = None
Height = None
scale = 0.00392

filename = "default"

# return output layer names
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw box around detected object
def draw_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) # get class label string
    color = COLORS[class_id] # get random color
    cv.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2) # create rectangle
    cv.putText(img, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # put text under rectangle
def yolo_frame(image):
    Width = image.shape[1]
    Height = image.shape[0]
    # create input blob as first step in dnn
    blob = cv.dnn.blobFromImage(image,scale, (416,416),(0,0,0), True, crop=False)    
    net.setInput(blob)# prepare input for neural network
    outputs = net.forward(get_output_layers(net))# call output layer names function
    class_ids = [] # initialize arrays for desired output layers (output that meets thresholds)
    confidences = []
    boxes = []
    conf_threshold = 0.6 # threshold for confidence in output objects
    nms_threshold = 0.3 # threshold for non-maximal suppression
    # loop through output layers checking whatever
    # label has highest confidence within layer, and keeping it based on thresholds
    for out in outputs:
        for detection in out:
            scores = detection[5:] # first five elements arent necessary
            class_id = np.argmax(scores) # classify as whatever object has highest confidence in this layer
            confidence = scores[class_id] # confidence variable
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # apply non maximal suppression to boxes array, leaves only maximal boxes
    # (boxes that are more likely to be single objects, not duplicates)
    indices = cv.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)
    # loop through suppressed boxes and draw them
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    cv.imshow(filename, image)# display final image

def yolo_video(fname):
    # read video
    vid = cv.VideoCapture(fname)
    
    assert vid is not None, "file could not be read, check with os.path.exists()"

    while(1): # loop through every single image frame and detect it
        ret, frame = vid.read()
        if ret == True:
            yolo_frame(frame)
            k = cv.waitKey(20) & 0xff
            if k == 10:
                break
        else:
            print("frames not read")
            break

parser = argparse.ArgumentParser(description='Simple implementation of YOLO for video input')
parser.add_argument('-p', '--path', default='slow_traffic_small.mp4')
args = parser.parse_args()

#prepare dnn

# read classes (objects the may be classified) from text file
classes = None
with open('yolov3.txt','r') as f:
    classes = [line.strip() for line in f.readlines()]

# use uniform random variable to generate colors for the boxes around classified objects
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# use readNet() in dnn to read pre trained weights file and configure the dnn
net = cv.dnn.readNet('yolov3.weights','yolov3.cfg')

filename = args.path
yolo_video(args.path)
print(args.path)
cv.destroyAllWindows()
