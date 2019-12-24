'''
View the first video
'''
import cv2

video_path = "/mnt/other/projects/GapWatch/week_1/6 sept Friday/MOVI0000.avi"



cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
while(1):
   ret, frame = cap.read()
   cv2.imshow('frame',frame)
   if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
       cap.release()
       cv2.destroyAllWindows()
       break
   cv2.imshow('frame',frame)