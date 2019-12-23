import requests
import numpy as np
import cv2

#TODO input this in a while loop
url = "http://10.150.0.37:5000/mask"
response = requests.get(url)
if response is not None:
    image = np.frombuffer(response.content, dtype=np.uint8)
    img = cv2.imdecode(image, flags=1)

    print(np.shape(img))
    print(img)

