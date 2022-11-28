__author__ = 'Vu Hoang Long'

import cv2
import yaml
import csv
import os
import numpy as np
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
#from utils.render import render
from utils.pose import get_pose
from Sim3DR import rasterize
from utils.tddfa_util import _to_ctype
import imgaug.augmenters as iaa
import random


def hflip(rgb, d):
    hflip= iaa.Fliplr(p=1.0)
    rgb = hflip.augment_image(rgb)
    try:
        d = hflip.augment_image(d)
    except:
        return rgb, d
    return rgb, d

def crop(rgb, d):
    crop1 = iaa.Crop(percent=(0, 0.3)) 
    rgb = crop1.augment_image(rgb)
    try:
        d = crop1.augment_image(d)
    except:
        return rgb, d
    return rgb, d

def add_contrast(rgb, d):
    rgb = iaa.Add(value=(-20,20), per_channel=True).augment_image(rgb)
    contrast = iaa.GammaContrast((0.5, 1.8))
    rgb = contrast.augment_image(rgb)
    return rgb, d

def nothing_aug2(rgb, d):
    return rgb, d


def random_rgb_depth_aug(rgb, d):
    my_list = [hflip, crop, add_contrast, add_contrast, crop]
    rgb, d = random.choice(my_list)(rgb, d)
    return rgb, d



ignore = ""

def crop_img(img, box, expand=0):
    h, w,_ = img.shape
    sx = int(round(box[0]))
    sy = int(round(box[1]))
    ex = int(round(box[2]))
    ey = int(round(box[3]))
    if expand > 0:
        if (sx >= expand and ex + expand <= w):
            sx -= expand
            ex += expand
        if (sy >= expand and ey + expand <= h):
            sy -= expand
            ey += expand
    return img[sy:ey, sx:ex]


class DepthMapData(object):
    def __init__(self, mode='cpu', onnx = False):
        self.dense_flag = 'depth'
        self.type = '.jpg'
        self.cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        if onnx:
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'
            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX
            self.face_boxes = FaceBoxes_ONNX()
            self.tddfa = TDDFA_ONNX(**self.cfg)
        else:
            gpu_mode = mode == 'gpu'
            self.tddfa = TDDFA(gpu_mode=gpu_mode, **self.cfg)
            self.face_boxes = FaceBoxes()


    def get_depth(self, img, bbox):
        param_lst, roi_box_lst = self.tddfa(img, bbox)
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=self.dense_flag)
        #show with origin background RGB
        #overlap = img.copy()
        overlap = np.zeros_like(img)
        for ver_ in ver_lst:
            ver = _to_ctype(ver_.T)  # transpose

            z = ver[:, 2]
            z_min, z_max = min(z), max(z)

            z = (z - z_min) / (z_max - z_min)

            # expand
            z = np.repeat(z[:, np.newaxis], 3, axis=1)

            overlap = rasterize(ver, self.tddfa.tri, z, bg=overlap)
        return overlap
    
    
    def get_face(self, image, label):
        #return x1, x2, y1, y2 and score of face
        try:
            box = self.face_boxes(image)
            face_img = crop_img(image, box[0], 0)
            param_lst, roi_box_lst = self.tddfa(image, box)
            yaw, pitch = get_pose(param_lst, roi_box_lst)
            if -35 < yaw < 35:
                face256 = cv2.resize(face_img, (256,256))
                bbox256 = self.face_boxes(face256)
                if label == '1':
                    #depth256 = self.get_depth(face256, bbox256)
                    return face256, None
                # path_save_rgb = output_rgb+name_img+self.type
                # path_save_d = output_d+name_img+self.type
                # self.save_rgb256(face256, path_save_rgb)
                # self.save_depth256(depth256, path_save_d)
                return face256, None
            else:
                print('Head face is not in the Middle!')
                return None, None
        except:
            print('Can not save RGB and depth map: Image is none!!')
            return None, None


def video_to_frames(pathIn='',
                    rgb_out='',
                    d_out='',
                    face_detector=None,
                    name_vid='',
                    label='',
                    png_compression=5):
    cap = cv2.VideoCapture(pathIn)
    #n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        success = True
        count = 0
        frame_count = 0
        name_frame = '{}_vid{}-fr'.format(label, name_vid)
        while success:
            success, image = cap.read()
            if success:
                print('ok')
                rgb256, d256 = face_detector.get_face(image=image,label=label)
                if frame_count < 50:
                    if rgb256 is not None:
                        if frame_count > 20:
                            rgb256, d256 = random_rgb_depth_aug(rgb256, d256)
                        #print(type(rgb256_aug), type(d256_aug))
                        if label == '1':
                            print('Write a new Face frame: {}, {}'.format(success, count + 1))
                            cv2.imwrite(os.path.join(rgb_out, "{}{:03d}.png".format(name_frame, count + 1)), rgb256,
                                        [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])  # save frame as PNG file
                            # gray_d256_aug = cv2.cvtColor(d256, cv2.COLOR_BGR2GRAY)
                            # print('Write a new Depth frame: {}, {}'.format(success, count + 1))
                            # cv2.imwrite(os.path.join(d_out, "{}d{:03d}.png".format(name_frame, count + 1)), gray_d256_aug,
                            #             [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])  # save frame as PNG file
                            count = count + 1
                            frame_count+=1
                        else:
                            print('Write a new Face frame: {}, {}'.format(success, count + 1))
                            cv2.imwrite(os.path.join(rgb_out, "{}{:03d}.png".format(name_frame, count + 1)), rgb256,
                                        [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])  # save frame as PNG file
                            count = count + 1
                            frame_count+=1
                else:
                    break
        cap.release()
    

def read_csv(csv_path):
    ''' 
    ''' 
    data_list = []
    csvFile = open(csv_path, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        data_list.append(item)
    return data_list



face_detect = DepthMapData(mode='cpu',onnx=False)

filenames = next(os.walk('/home/long/Zalo22-VHL/video/videos'), (None, None, []))[2]
#print(filenames)
rgb_out_dir = '/home/long/Zalo22-VHL/data/rgb/'
d_out_dir = '/home/long/Zalo22-VHL/data/d/'
label_csv = read_csv('/home/long/Zalo22-VHL/video/label.csv')
vids_path = '/home/long/Zalo22-VHL/video/videos/'
for i in range(1, len(label_csv)):
    vid_name = label_csv[i][0].split('.')
    vid_file = vids_path + vid_name[0] + '.' + vid_name[1]
    #print(vid_file)
    print(vid_file)
    label = label_csv[i][1] 
    print(label)
    video_to_frames(vid_file, rgb_out_dir, d_out_dir, face_detector=face_detect, name_vid=vid_name[0], label=label)
    