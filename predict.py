from preprocessing import DepthMapData
from model import RevisitResNet50
import os
from process_predict import preprocess
import tensorflow as tf
import csv
import numpy as np
import argparse
import time


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
saved_model_path = os.path.join(ROOT_DIR, 'saved_models/cp.ckpt')
print(saved_model_path)
result_dir = os.path.join(ROOT_DIR, 'results')


#Create result folder: ../results/
if not os.path.exists(result_dir):
      os.makedirs(result_dir)

#Creare model for predicting
revisit_model = RevisitResNet50()
revisit_model.build_model()
revisit_model.load_weights(saved_model_path)
#revisit_model.summary()


parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--test_path', type=str,
                    help='Path to videos testset')

args = parser.parse_args()

result_f = open(os.path.join(result_dir,'submission.csv'), 'w')
result_t = open(os.path.join(result_dir,'time_submission.csv'), 'w')
writer_f = csv.writer(result_f)
writer_t = csv.writer(result_t)
# print(vids_path)

#Define result file
writer_f.writerow(['fname', 'liveness_score'])
writer_t.writerow(['fname', 'time'])

#Get all test videos path
all_video_path = os.path.join(ROOT_DIR,args.test_path,'videos')
vid_list = os.listdir(all_video_path)


th = 0.06
for vid in vid_list:
    vid_path = os.path.join(ROOT_DIR, vid)
    face_detect = False
    frame_count = 0
    pred_all = []
    time_all = []
    pred_all.append(vid)
    time_all.append(vid)
    start = time()    
    tensor_input = preprocess(video_path=vid_path)
    pred_label = tf.get_static_value(revisit_model.predict(tensor_input)[0][0])
    if pred_label > th:
      pred_label = 1
    else:
      pred_label = 0
    end = time()
    diff = int(start*1000 - end*1000)
    pred_all.append(str(pred_label))
    time_all.append(str(diff))
    writer_f.writerow(pred_all)
    writer_t.writerow(time_all)
    # print('Total frames: ', n_frames)
    # print('Num frames detect Face: ', frame_count)
    print('Video name: ', vid)
    print('Label: ', pred_label)    
result_f.close()