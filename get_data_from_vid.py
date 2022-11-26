import cv2
import csv
import os

def video_to_frames(pathIn='',
                    pathOut='',
                    face_detector=None,
                    name_vid='',
                    label='',
                    png_compression=9):
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
                if face_detector is None:
                    #image = face_detector.face_detect(img=image)
                    if image is not None and frame_count <= 10:
                        print('Write a new Face frame: {}, {}'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}{:03d}.png".format(name_frame, count + 1)), image,
                                    [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])  # save frame as PNG file
                        count = count + 1
                        frame_count+=1
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


filenames = next(os.walk('/home/long/3DDFA_V2/video/videos'), (None, None, []))[2]
print(filenames)
images_out_dir = '/home/long/3DDFA_V2/data/rgb'
label_csv = read_csv('/home/long/3DDFA_V2/video/label.csv')
vids_path = '/home/long/3DDFA_V2/video/videos/'
#face_detector = FaceDetection()
for i in range(1, len(label_csv)):
    vid_name = label_csv[i][0].split('.')
    vid_file = vids_path + vid_name[0] + '.' + vid_name[1]
    #print(vid_file)
    label = label_csv[i][1] 
    video_to_frames(vid_file, images_out_dir, name_vid=vid_name[0], label=label)
    print(vid_file)