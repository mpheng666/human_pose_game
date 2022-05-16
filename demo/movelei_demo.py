from turtle import left
import tensorflow as tf
import cv2
import numpy as np
from pynput.keyboard import Key, Controller
import time

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

class MoveNetInference:
    def __init__(self):
        self.model_path = "../human_pose_estimation/model/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite" 
        self.media_path = "media/how-to-squat.jpg" 

        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.confidence_thresold = .3
        self.tf_size = (192, 192)
        self.success = True
        self.keyboard = Controller()
        self.counter = 0
        self.counter_left = 0
        self.counter_right = 0
        self.counter_left_percent = 0

    def OpenCamera(self):
        self.default_video_source = 0
        self.cap = cv2.VideoCapture(self.default_video_source)

        if not self.cap.isOpened():
            print('Error loading video')
            quit()

        self.success, self.img = self.cap.read()

        if not self.success:
            print('Error reding frame')
            quit()

        self.y, self.x, self._ = self.img.shape

    def ReadImage(self):
        self.img = cv2.imread(self.media_path)
        self.y, self.x, self._ = self.img.shape

    def PrepareImage(self):
        self.tf_img = cv2.resize(self.img, self.tf_size)
        self.tf_img = cv2.cvtColor(self.tf_img, cv2.COLOR_BGR2RGB)
        self.tf_img = np.asarray(self.tf_img)
        self.tf_img = np.expand_dims(self.tf_img,axis=0)

    def StartInferencing(self):
        self.OpenCamera()
    
        while self.success:
            # self.ReadImage()
            self.PrepareImage()
            self.input_image = tf.cast(self.tf_img, dtype=tf.uint8)

            # TF Lite format expects tensor type of float32.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.interpreter.set_tensor(self.input_details[0]['index'], self.input_image.numpy())
            self.interpreter.invoke()

            # Output is a [1, 1, 17, 3] numpy array.
            self.keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])

            # iterate through keypoints
            # print(self.keypoints_with_scores.shape)
            for k in self.keypoints_with_scores[0,0,:,:]:
                # Converts to numpy array
                # print(self.keypoints_with_scores.shape)
                # print(k.shape)

                # Checks confidence for keypoint
                if k[2] > self.confidence_thresold:
                    # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
                    yc = int(k[0] * self.y)
                    xc = int(k[1] * self.x)

                    self.counter += 1

                    if (k[1] >= 0.5):
                        colour = (0, 255, 0)
                        self.counter_left += 1
                    else:
                        colour = (0, 0, 255)

                    # Draws a circle on the image for each keypoint
                    self.img = cv2.circle(self.img, (xc, yc), 2, colour, 5)
                    self.flipverticalimg = cv2.flip(self.img, 1)
            
                    # Shows image
                    cv2.imshow('Movelei', self.flipverticalimg)

            if(self.counter == 0):
                self.counter = 1

            self.counter_left_percent = self.counter_left/self.counter
            print(self.counter_left_percent)

            self.keyboard.release(Key.left)          
            self.keyboard.release(Key.right)
            
            if(self.counter_left_percent > 0.5):  
                self.keyboard.press(Key.left)
                # time.sleep(0.01)
                # for x in range(100):
                #     # self.keyboard.press(Key.left)
                #     # self.keyboard.release(Key.left)
                #     x += 1
            else:
                self.keyboard.press(Key.right)
                # time.sleep(0.01)
                # for x in range(100):
                #     # self.keyboard.press(Key.right)
                #     # self.keyboard.release(Key.right)
                #     x += 1            

            self.counter = 0
            self.counter_left = 0
            self.counter_right = 0
            # Waits for the next frame, checks if q was pressed to quit
            if cv2.waitKey(1) == ord("q"):
                break

            # Reads next frame
            self.success, self.img = self.cap.read()

        self.cap.release()

    def _keypoints_and_edges_for_display(keypoints_with_scores,
                                        height,
                                        width,
                                        keypoint_threshold=0.11):
        """Returns high confidence keypoints and edges for visualization.

        Args:
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.
            height: height of the image in pixels.
            width: width of the image in pixels.
            keypoint_threshold: minimum confidence score for a keypoint to be
            visualized.

        Returns:
            A (keypoints_xy, edges_xy, edge_colors) containing:
            * the coordinates of all keypoints of all detected entities;
            * the coordinates of all skeleton edges of all detected entities;
            * the colors in which the edges should be plotted.
        """
        keypoints_all = []
        keypoint_edges_all = []
        edge_colors = []
        num_instances, _, _, _ = keypoints_with_scores.shape
        for idx in range(num_instances):
            kpts_x = keypoints_with_scores[0, idx, :, 1]
            kpts_y = keypoints_with_scores[0, idx, :, 0]
            kpts_scores = keypoints_with_scores[0, idx, :, 2]

            kpts_absolute_xy = np.stack(
                [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)

            kpts_above_thresh_absolute = kpts_absolute_xy[
                kpts_scores > keypoint_threshold, :]

            keypoints_all.append(kpts_above_thresh_absolute)

            for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
                if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                    x_start = kpts_absolute_xy[edge_pair[0], 0]
                    y_start = kpts_absolute_xy[edge_pair[0], 1]
                    x_end = kpts_absolute_xy[edge_pair[1], 0]
                    y_end = kpts_absolute_xy[edge_pair[1], 1]
                    line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                    keypoint_edges_all.append(line_seg)
                    edge_colors.append(color)

        if keypoints_all:
            keypoints_xy = np.concatenate(keypoints_all, axis=0)
        else:
            keypoints_xy = np.zeros((0, 17, 2))

        if keypoint_edges_all:
            edges_xy = np.stack(keypoint_edges_all, axis=0)
        else:
            edges_xy = np.zeros((0, 2, 2))
        return keypoints_xy, edges_xy, edge_colors

    # def SentKey

    def DrawPointsandLines(self):
        pass

    def ShowImage(self):
        pass
        
if __name__ == "__main__":
    human_pose_inference_1 = MoveNetInference()
    human_pose_inference_1.StartInferencing()