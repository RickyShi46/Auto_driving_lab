# This server used to receive images and current vehicle data from the client, passes the images to a trained classifier to predict new vehicle commands, 
# and then sends these commands back to Unity to control the car for autonomous driving.

# now we only test the communication between the client and server at first, 
# so we just give the fixed command to see wether the car will be controlled and wether the images succesffly recieved
# lately we can also add code to realize autonomous driving

import socketio
# concurrent networking
import eventlet
# web server gateway interface
import eventlet.wsgi
from flask import Flask
import base64
import cv2
import numpy as np
# from io import BytesIO
import time
import os
import torch
from networks.resnet import resnet_model
from safety_distance import SafetyDistance
from yolov5 import Yolov5Detector as Yolov5

# Initialize Socket.IO server
sio = socketio.Server()
app = Flask(__name__)

frame_count = 0
frame_count_save = 0
prev_time = 0
fps = 0

model = torch.load('/home/zyj/Praktikum/autonomous-driving-simulator/weights/resnet/1.pt')
#model = torch.load('/home/zyj/Praktikum/1autonomous-driving-simulator/weights/resnet/0.pth')

# Validation of yolo class and initialisation of Yolo
detector = Yolov5.Yolov5Detector()
stride, names, pt, jit, onnx, engine, imgsz, device = detector.detectorInit()
detector.detectorWarmUp()
print("Yolov5Detector is ready!")


@sio.on("send_image")
def on_image(sid, data):
    #make the variables global to calculate the fps 
    global frame_count, frame_count_save, prev_time, fps
    #print("image recieved!")
    img_data = data["image"]
    img_bytes = base64.b64decode(img_data)
    # Decode image from base64 format，将字典里抽取出来的字符串转换为字节串类型
    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # # create a variable to identify every frame of image for the lately image-save
    # frame_count_save += 1

    # Calculate and print fps
    frame_count += 1
    elapsed_time = time.time() - prev_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        prev_time = time.time()
        frame_count = 0

    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        img = img.transpose((2, 0, 1)) 
        img = np.expand_dims(img, axis=0)  

        img = torch.from_numpy(img).float()  
        # predictions = model(img) 
        # print(predictions.size())
        # print(predictions)
        # brakes = float(predictions[0, 1])
        # steering_angle = float(predictions[0, 2])
        # throttle = float(predictions[0, 0])
        # print("计算出的steering_angle:",steering_angle)
        # print("计算出的throttle:",throttle)
        # print("计算出的brakes:",brakes)
        # send_control(steering_angle, throttle, brakes)
        # print("成功发送指令")
        # send to yolo to detect the object in image and return processed image

        brakes = 0
        steering_angle = 0
        throttle = 0
        cv_ImgRet, detect, _ = detector.detectImage(
            img[0], img[0].transpose(1, 2, 0), needProcess=True)
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
                brakes = 1
            if 'warning' in flags:  
                brakes = 0.5 
                

        send_control(steering_angle, throttle, brakes)
        print("成功发送指令")

    else:
        print("Invalid image data")


# listen for the event "vehicle_data"
@sio.on("vehicle_data")
def vehicle_command(sid, data):
    # print("data recieved!")
    steering_angle = float(data["steering_angle"])
    throttle = float(data["throttle"])
    # brakes = float(data["brakes"])
    print("接收到了车辆目前的信息")
    print("目前的steering_angle:",steering_angle)
    print("目前的throttle:",throttle)

    if data:
        # steering_angle = 0
        # throttle = 0.3

        send_control(steering_angle, throttle, 0)
    else:
        # send the data to unityClient
        # sio.emit("manual", data={})
        print("data is empty")


@sio.event
def connect(sid, environ):
    # sid for identifying the client connected表示客户端唯一标识符，environ表示其连接的相关环境信息
    print("Client connected")
    send_control(0, 0, 0)

# Define a data sending function to send processed data back to unity client
def send_control(steering_angle, throttle, brakes):
    print("发送控制数据")
    sio.emit(
        "control_command",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
            "brakes": brakes.__str__(),
        },
        skip_sid=True,
    )

@sio.event
def disconnect(sid):
    # implement this function, if disconnected
    print("Client disconnected")


app = socketio.Middleware(sio, app)
# Connect to Socket.IO client
if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)