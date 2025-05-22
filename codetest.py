import os
import glob
import shutil

class_list = glob.glob(os.path.join('data', 'train', '*'))

class_dict = {}
for idx, class_file in enumerate(class_list):
    class_name = os.path.basename(class_file)
    class_dict[class_name] = idx

    class_imgs = glob.glob(os.path.join(class_file, '*.jpg'))
    for index, img in enumerate(class_imgs):
        new_name = os.path.join(class_file, f'{index}.jpg')
        shutil.move(img, new_name)

    shutil.move(class_file, os.path.join('data', 'train', str(idx)))

with open('class_dict.txt', 'w') as f:
    for class_name, idx in class_dict.items():
        f.write(f'{class_name} {idx}\n')
    