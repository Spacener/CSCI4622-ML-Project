# ------------------------------------------------
# Team LearnCraft
# 1/20/22
#
# main.py
# ------------------------------------------------


import cv2
import win32gui
import time
from protocols.eyes import readInactive
from protocols.legs import onward

hwnd = win32gui.FindWindow(None, "Minecraft 1.18.1 - Singleplayer")
img, left, top = readInactive(hwnd)

for i in list(range(2)):
    print(i+1)
    time.sleep(1)

last_time = time.time()
playerLock = 0
invenUp = 0

while 1:
    print("[DEBUG]: Stream latency: {} seconds".format(time.time()-last_time))
    last_time = time.time()

    img, left, top = readInactive(hwnd) # read screen!

    # onward()
    print(img)
    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        break

    print("[DEBUG]: Stream latency: {} seconds".format(time.time() - last_time))
    last_time = time.time()