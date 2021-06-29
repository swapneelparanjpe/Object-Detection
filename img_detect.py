import cv2
import numpy as np
import time
import sys

def detect_objects_in_image(filename):
    s = time.time()

    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes=[]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]              # creates a list of all classes in coco.names
    layer_names = net.getLayerNames()                                   # returns a list of all layer names
    yolo_layers_index = net.getUnconnectedOutLayers()                   # returns indices of yolo layers
    output_layers = [layer_names[i[0]-1] for i in yolo_layers_index]    # creates list of yolo layers using above indices

    img = cv2.imread(filename)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, False)

    # Passing the blob image to YOLO model
    net.setInput(blob)
    output = net.forward(output_layers) 
    # output contains all information in the image

    # Displaying information on screen
    boxes = []
    confidences = []
    class_ids = []
    for out in output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >= 0.5:
                x = int(detection[0]*width)
                y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # Rectangle coordinates
                x1 = int(x - (w/2))
                y1 = int(y - (h/2))
                
                boxes.append((x1,y1,w,h))
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Applying non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4 )

    # Putting bounding boxes and label
    no_objects_detected = len(boxes)
    for i in range(no_objects_detected):
        if i in indices:
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]
            color = (0, 255, 0)
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(img, label, (x+30, y-30), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
            
    print("Time taken:", time.time()-s)
    cv2.imshow("Output detection", img)
    cv2.imwrite("img_detection_output.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    filename = sys.argv[1]
    detect_objects_in_image(filename)