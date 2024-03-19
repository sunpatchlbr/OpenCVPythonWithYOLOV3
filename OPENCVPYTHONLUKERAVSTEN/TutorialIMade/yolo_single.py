import cv2 as cv
import numpy as np
import matplotlib as plt

# read single image
fname = 'cafe.jpeg'
image = cv.imread(fname, cv.COLOR_RGB2BGR)
assert image is not None, "file could not be read, check with os.path.exists()"

Width = image.shape[1]
Height = image.shape[0]
scale = 0.003

# read classes (objects the may be classified) from text file
classes = None
with open('yolov3.txt','r') as f:
    classes = [line.strip() for line in f.readlines()]

# use uniform random variable to generate colors for the boxes around classified objects
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# use readNet() in dnn to read pre trained weights file and configure the dnn
net = cv.dnn.readNet('yolov3.weights','yolov3.cfg')

# create input blob as first step in dnn
# parameters:
# target image
# scalar used on values in image
# size of output image, must match yolo v3 output layer size
# boolean to swap r and b channel
blob = cv.dnn.blobFromImage(image,scale, (416,416),(0,0,0), True, crop=False)

# prepare input for neural network
net.setInput(blob)

# return output layer names
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw box around detected object
def draw_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + ' ' + str(round(confidence,2)) # get class label string
    color = COLORS[class_id] # get random color
    cv.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2) # create rectangle
    cv.putText(img, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # put text under rectangle

# call output layer names function
outputs = net.forward(get_output_layers(net))

# initialize arrays for desired output layers (output that meets thresholds)
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.7 # threshold for confidence in output objects
nms_threshold = 0.4 # threshold for non-maximal suppression

# loop through output layers checking whatever label has
# highest confidence within layer, and keeping it based on thresholds
for out in outputs:
    for detection in out:
        scores = detection[5:] # first five elements arent necessary
        class_id = np.argmax(scores) # classify as max confidence in layer
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

# apply non maximal suppression to boxes array, leaves only maximal
# boxes (boxes that are more likely to be single objects, not duplicates)
indices = cv.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)

# loop through suppressed boxes and draw them
for i in indices:
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    draw_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# display final image
cv.imshow(fname, image)

cv.waitKey(0)

cv.destroyAllWindows()
