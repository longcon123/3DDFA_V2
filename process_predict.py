import cv2 
import tensorflow as tf
import numpy as np
from preprocessing import DepthMapData


face_detector = DepthMapData()


def img2tf(image):
  #image = cv2.resize(image, (256,256))
  image = tf.convert_to_tensor(image, dtype=tf.float32)
  image = tf.image.per_image_standardization(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = np.expand_dims(image, axis=0)
  return image

def preprocess(video_path):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    face_detect = False
    while(face_detect==False and frame_count < n_frames):
        ret, frame = cap.read()
        frame_count+=1
        if ret == True:
            face_frame,_ = face_detector.get_face(frame, '0')
            if face_frame is not None:
                #cv2_imshow(face_frame)
                face_detect=True
                return img2tf(face_frame)
    cap.release()
     