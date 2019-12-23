# import the necessary packages
import numpy as np
import time
import urllib.request
import cv2

host_url = 'http://0.0.0.0:5000/video_feed_dpt'

stream = urllib.request.urlopen(host_url)
time.sleep(1)

bytes = b''
while True:
    bytes += stream.read(2**14)

    a = bytes.find(b'\xff\xd8\xf1')
    b = bytes.find(b'\xff\xd9\xf2')

    if a != -1 and b != -1:
        data = bytes[a+3:b]
        bytes = bytes[b+3:]

        arr = np.frombuffer(data, dtype=np.float32)
        arr = np.reshape(arr, (76800,3))

        time_start=time.time()
        if cv2.waitKey(1) == 27:
            exit(0)





