import cv2
import numpy as np
# this is the "yolo.py" file, I assume it's in the same folder as this program
from yolo import Yolo
count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)

# these are the filepaths of the yolo files
weights = "yolov3-tiny.weights";
config = "yolov3-tiny.cfg";
labels = "yolov3.txt";

# init yolo network
target_class_id = 79; # toothbrush
conf_thresh = 0.4; # less == more boxes (but more false positives)
nms_thresh = 0.4; # less == more boxes (but more overlap)
net = Yolo(config, weights, labels, conf_thresh, nms_thresh, use_cuda = True);

# open video capture
cap = cv2.VideoCapture(0); # probably laptop webcam

# loop
done = False;
while not done:
    # get frame
    ret, frame = cap.read();
    if not ret:
        done = cv2.waitKey(1) == ord('q');
        continue;

    # draw detection
    # frame, _ = net.detect(frame, target_id=target_class_id); # use this to filter by a single class_id
    frame, _ = net.detect(frame); # use this to not filter by class_id

    # show
    cv2.imshow("Marked", frame);
    done = cv2.waitKey(1) == ord('q');
