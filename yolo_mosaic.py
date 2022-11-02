import cv2
import numpy as np

class yolo_mosaic():
    # Initialize parameters
    def __init__(self, net_filename, conf_threshold=0.5, nms_threshold=0.5, obj_threshold=0.5):
        anchors = [[4, 5,  8, 10,  13, 16], [23, 29,  43,
                                             55,  73, 105], [146, 217,  231, 300,  335, 433]]
        num_classes = 1
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5 + 10
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(
            anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inp_width = 640
        self.inp_height = 640
        self.net = cv2.dnn.readNet(net_filename)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.obj_threshold = obj_threshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def post_process(self, frame, outs):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        ratioh, ratiow = frame_height / self.inp_height, frame_width / self.inp_width

        # Scan through all the bounding boxes output from the network and keep only the ones with
        # high confidence scores. Assign the box's class label as the class with the highest score.
        confidences = []
        boxes = []
        landmarks = []
        for detection in outs:
            confidence = detection[15]
            if detection[4] > self.obj_threshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                landmark = detection[5:15] * \
                    np.tile(np.float32([ratiow, ratioh]), 5)
                landmarks.append(landmark.astype(np.int32))

        # Perform non maximum suppression to eliminate redundant overlapping boxes with lower
        # confidences.
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            landmark = landmarks[i]
            frame = self.draw_pred(
                frame, left, top, left + width, top + height, landmark)
        return frame

    # Add mosaic to the image crop
    def crop_mosaic(self, img, alpha):
        crop_width = img.shape[1]
        crop_height = img.shape[0]

        # Add mosaic by resizing the picture (INTER_LINEAR or INTER_NEAREST)
        img = cv2.resize(img, (int(crop_width*alpha + 1), int(crop_height*alpha + 1)))
        img = cv2.resize(img, (crop_width, crop_height),
                         interpolation=cv2.INTER_LINEAR)

        return img

    # Draw mosaic on predicted face(s)
    def draw_pred(self, frame, left, top, right, bottom, landmark):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Pre-process, to handle unexpected prediction
        top = 0 if top < 0 else top
        top = frame_height if top > frame_height else top
        bottom = 0 if bottom < 0 else bottom
        bottom = frame_height if bottom > frame_height else bottom
        left = 0 if left < 0 else left
        left = frame_width if left > frame_width else left
        right = 0 if right < 0 else right
        right = frame_width if right > frame_width else right

        # Draw mosaic on the predicted face
        frame[top:bottom, left:right] = self.crop_mosaic(
            frame[top:bottom, left:right], 0.02)

        return frame

    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(
            srcimg, 1 / 255.0, (self.inp_width, self.inp_height), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        # Inference output
        outs[..., [0, 1, 2, 3, 4, 15]] = 1 / \
            (1 + np.exp(-outs[..., [0, 1, 2, 3, 4, 15]]))   # sigmoid function
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inp_height /
                       self.stride[i]), int(self.inp_width/self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)

            g_i = np.tile(self.grid[i], (self.na, 1))
            a_g_i = np.repeat(self.anchor_grid[i], h * w, axis=0)
            outs[row_ind:row_ind + length, 0:2] = (
                outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + g_i) * int(self.stride[i])
            outs[row_ind:row_ind + length,
                 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * a_g_i

            outs[row_ind:row_ind + length, 5:7] = outs[row_ind:row_ind + length,
                                                       5:7] * a_g_i + g_i * int(self.stride[i])   # landmark x1 y1
            outs[row_ind:row_ind + length, 7:9] = outs[row_ind:row_ind +
                                                       length, 7:9] * a_g_i + g_i * int(self.stride[i])  # landmark x2 y2
            outs[row_ind:row_ind + length, 9:11] = outs[row_ind:row_ind +
                                                        length, 9:11] * a_g_i + g_i * int(self.stride[i])  # landmark x3 y3
            outs[row_ind:row_ind + length, 11:13] = outs[row_ind:row_ind +
                                                         length, 11:13] * a_g_i + g_i * int(self.stride[i])  # landmark x4 y4
            outs[row_ind:row_ind + length, 13:15] = outs[row_ind:row_ind +
                                                         length, 13:15] * a_g_i + g_i * int(self.stride[i])  # landmark x5 y5
            row_ind += length

        return outs