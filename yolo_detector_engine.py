import cv2
import numpy as np
from safety_distance import SafetyDistance
from yolov5 import Yolov5Detector as Yolov5



class Script(EngineScript):
    def initialize(self):

        # Validation of yolo class and initialisation of Yolo
        self.detector = Yolov5.Yolov5Detector()
        stride, names, pt, jit, onnx, engine, imgsz, device = self.detector.detectorInit()
        self.detector.detectorWarmUp()
        print("Yolov5Detector is ready!")

    def runLoop(self, timestep_ns):
        # Requesting the image data from camera_tf Transceiver Function
        image_Info = self._getDataPack("camera_img")
        img_list = image_Info['current_image_frame']
        img_width = image_Info['c_imageWidth']
        img_height = image_Info['c_imageHeight']

        detect = ()
        # Judgment of getting image data
        if img_width == 736:

            # reshape to proper size
            img_frame = np.array(img_list, dtype=np.uint8)
            cv_image = img_frame.reshape((img_height, img_width, 3))
            # Convertion from RGB to BGR
            cv_image = cv_image[:, :, ::-1] - np.zeros_like(cv_image)
            # Convertion to normal form of the numpy image
            np_image = cv_image.transpose(2, 0, 1)

            # send to yolo to detect the object in image and return processed image
            cv_ImgRet, detect, _ = self.detector.detectImage(
                np_image, cv_image, needProcess=True)
            # Create a safe distance object
            safety_distance = SafetyDistance(150, 250)

            # Get the coordinates and category of all the detection boxes
            boxs = [d[:4] for d in detect]
            classes = [d[5] for d in detect]

            # Calculate safety distances
            distance = safety_distance.compute_distance(
                cv_ImgRet, list(zip(classes, boxs)))
            safety_distance.process(cv_ImgRet, list(zip(classes, boxs)))
            results = safety_distance.process(
                cv_ImgRet, list(zip(classes, boxs)))
            safety_distance.draw_distance(cv_ImgRet, results)
            if results:
                flags = [result[4] for result in results]
                # If it is dangerous, then the coordinates of the detection frame have to be passed to a function that converts the position of the detection frame to world coordinates and then passes this position to the vehicle control function
                if 'danger' in flags:
                    state = self._getDataPack("state_location")
                    if boxs:
                        print(boxs)
                        obstacle_position = YOLOv5CameraTransformer. transform(
                            boxs, state)
                        print(obstacle_position)
                        self._setDataPack("obstacle_position", {
                                          "obstacle_x": obstacle_position[0], "obstacle_y": obstacle_position[1]})
        else:
            # If getting no images, set the window black
            cv_ImgRet = np.zeros((200, 300, 3), dtype=np.uint8)

        # OpenCV shows the current image frame
        cv2.imshow('detected image', cv_ImgRet)
        cv2.waitKey(1)

        # d = np.frombuffer(img_frame, np.uint8)

    def shutdown(self):
        print("Yolo detector engine is shutting down")
