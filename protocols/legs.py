
# Onward!

from time import sleep
import pydirectinput

def onward():
    pydirectinput.keyDown('w')

def whence():
    pydirectinput.keyDown('w')

def portside():
    pydirectinput.keyDown('a')

def starboard():
    pydirectinput.keyDown('a')

def halt():
    pydirectinput.keyUp('w')
    pydirectinput.keyUp('a')
    pydirectinput.keyUp('s')
    pydirectinput.keyUp('d')