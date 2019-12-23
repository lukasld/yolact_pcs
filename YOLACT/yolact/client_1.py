import cv2
import imagezmq

image_hub = imagezmq.ImageHub()
while True:
    cam_name, image =image_hub.recv_image()
    print(image)
    image_hub.send_reply(b'400')
