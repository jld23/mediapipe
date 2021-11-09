import cv2
import numpy as np

class Yolo:
    def __init__(self, cfg, weights, names, conf_thresh, nms_thresh, use_cuda = False):
        # save thresholds
        self.ct = conf_thresh;
        self.nmst = nms_thresh;

        # create net
        self.net = cv2.dnn.readNet(weights, cfg);
        print("Finished: " + str(weights));
        self.classes = [];
        file = open(names, 'r');
        for line in file:
            self.classes.append(line.strip());

        # use gpu + CUDA to speed up detections
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA);
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA);

        # get output names
        layer_names = self.net.getLayerNames();
        self.output_layers = [layer_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()];

    # runs detection on the image and draws on it
    def detect(self, img, target_id = None):
        # get detection stuff
        b, c, ids, idxs = self.get_detection_data(img, target_id);

        # draw result
        img = self.draw(img, b, c, ids, idxs);
        return img, len(idxs);

    # returns boxes, confidences, class_ids, and indexes (indices?)
    def get_detection_data(self, img, target_id = None):
        # get output
        layer_outputs = self.get_inf(img);

        # get dims
        height, width = img.shape[:2];

        # filter thresholds and target
        b, c, ids, idxs = self.thresh(layer_outputs, width, height, target_id);
        return b, c, ids, idxs;

    # runs the network on an image
    def get_inf(self, img):
        # construct a blob
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416,416), swapRB=True, crop=False);

        # get response
        self.net.setInput(blob);
        layer_outputs = self.net.forward(self.output_layers);
        return layer_outputs;

    # filters the layer output by conf, nms and id
    def thresh(self, layer_outputs, width, height, target_id = None):
        # some lists
        boxes = [];
        confidences = [];
        class_ids = [];

        # each layer outputs
        for output in layer_outputs:
            for detection in output:
                # get id and confidence
                scores = detection[5:];
                class_id = np.argmax(scores);
                confidence = scores[class_id];

                # filter out low confidence
                if confidence > self.ct:
                    # filter by target_id if set
                    if target_id is None or class_id == target_id:
                        # scale bounding box back to the image size
                        box = detection[0:4] * np.array([width, height, width, height]);
                        (cx, cy, w, h) = box.astype('int');

                        # grab the top-left corner of the box
                        tx = int(cx - (w / 2));
                        ty = int(cy - (h / 2));

                        # update lists
                        boxes.append([tx,ty,int(w),int(h)]);
                        confidences.append(float(confidence));
                        class_ids.append(class_id);

        # apply NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.ct, self.nmst);
        return boxes, confidences, class_ids, idxs;

    # draw detections on image
    def draw(self, img, boxes, confidences, class_ids, idxs):
        # check for zero
        if len(idxs) > 0:
            # loop over indices
            for i in idxs.flatten():
                # extract the bounding box coords
                (x,y) = (boxes[i][0], boxes[i][1]);
                (w,h) = (boxes[i][2], boxes[i][3]);

                # draw a box
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2);

                # draw text
                text = "{}: {:.4}".format(self.classes[class_ids[i]], confidences[i]);
                cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2);
        return img;
