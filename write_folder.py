import os
import shutil


image_path = r'/home/long/3DDFA_V2/data/rgb/'
old_path = r'/home/long/3DDFA_V2/data/rgb/'
new_path = r'/home/long/3DDFA_V2/data/Zalo22_val/rgb_by_folder/'

count = 368
def write_image2folder(images_path, old_path, new_path):
    count = 0
    list_images =  sorted((os.listdir(images_path)))
    for vid in list_images:
        name = vid.split("-")[0]
        label = name.split("_")[0]
        new_folder_name = name.split("_")[1] + "_" + label
        new_folder = new_path + new_folder_name
        print(label)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            count+=1
        os.replace(old_path+vid, new_path+new_folder_name+'/'+vid)
        if count == 369:
            break


# write_image2folder(images_path=image_path, old_path=old_path, new_path=new_path)

need_path = '/home/long/3DDFA_V2/Zalo22_DepthMap_Valset/rgb_by_folder/'
new_path = '/home/long/3DDFA_V2/Zalo22_DepthMap_Valset/d'
d_file = '/home/long/3DDFA_V2/Zalo22_DepthMap_Dataset/d_by_folder/'
def move2other():
    count = 0
    list_rgb =  sorted(os.listdir(need_path))
    print(len(list_rgb))
    for file_rgb in list_rgb:
        d = file_rgb.split('_')
        vid = d[0]
        label = d[1]
        if label == '1':
            count+=1
            try:
                shutil.move(d_file+vid, new_path)
            except:
                print('Exist:', new_path)


move2other()
#shutil.move(d_file+'vid1554', new_path)