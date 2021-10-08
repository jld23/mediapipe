import timeit
from pathlib import Path
import cv2

count = cv2.cuda.getCudaEnabledDeviceCount()
print("GPUs found:{}".format(count))

video = Path('AV9_14.MOV')

def run():

    video_capture = cv2.VideoCapture(str(video))
    last_ret = None
    last_frame = None
    while video_capture.get(cv2.CAP_PROP_POS_FRAMES) < video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
        ret, frame = video_capture.read()
        if last_ret is None or last_ret != ret:
            last_ret = ret
            print("Return code:{}".format(ret))

        if last_frame is None or last_frame.shape != frame.shape:
            last_frame = frame
            print("Frame Shape:{}".format(frame.shape))

timer = timeit.timeit('run()', number=10, setup="from __main__ import run")
print("time to read video 10x :{}".format(timer))
