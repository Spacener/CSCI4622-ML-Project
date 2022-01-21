
# Contains different types of image processing, for different tasks
# "What am I looking at"

import cv2

from protocols import legs
import numpy as np
import imutils
import win32gui
import win32ui
import win32.lib.win32con as win32con


# reads inactive window as np array
def readInactive(hwnd):
    # left, top, right, bot = win32gui.GetClientRect(hwnd)
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top

    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(saveBitMap)

    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(w, h) , dcObj, (0,0), win32con.SRCCOPY)

    im = dataBitMap.GetBitmapBits(False)
    img = np.array(im).astype(dtype="uint8")
    img.shape = (h, w, 4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("screenshot.png", img)
    # dataBitMap.SaveBitmapFile(cDC, 'screenshot.bmp')
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return img, left, top