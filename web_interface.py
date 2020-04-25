# web_interface.py
# Creates html file to stream video feed to web browser.

import flask, cv2, datetime, threading, argparse
from MotionDetector import MotionDetector



