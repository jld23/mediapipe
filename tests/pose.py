import cv2
import mediapipe as mp
import time
from pathlib import Path

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

vid = Path('AV 9_14.MOV')
pTime = 0

lm_list = []
# For webcam input:
cap = cv2.VideoCapture(str(vid))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_fps = round(cap.get(cv2.CAP_PROP_FPS))
pose = mp_pose.Pose(min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5, 
                    static_image_mode = False, 
                    model_complexity = 2,)

out_filename = ''.join([str(vid.parent), vid.anchor, vid.stem, '_annotated', '.avi'])
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'DIVX'), vid_fps, (frame_width,frame_height))
# while cap.isOpened():
while cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
  success, image = cap.read()
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  # print(fps)
  # convert the BGR image to RGB.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  
  if not success:
    break
    
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  pose_start = time.time()
  results = pose.process(image)
  pose_time = time.time() - pose_start
  if results.pose_landmarks:
    cv2.putText(image, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.putText(image, str(round(pose_time,4)), (70,150), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    out.write(image)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c  = image.shape
        lm_list.append([id, lm.x, lm.y, lm.z, lm.visibility])
        # cv2.imshow('Pose Estimation', cv2.resize(image, (frame_width,frame_height)))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
out.release()
