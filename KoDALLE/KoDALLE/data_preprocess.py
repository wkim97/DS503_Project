from glob import glob
from tqdm import tqdm
import os, json
from pathlib import Path
from shutil import copy2

base_path = '/mnt/hdd1/wkim/DS503/KoDALLE/data/montage/data/train/labeled_data'
json_list = glob(os.path.join(base_path, 'json/*/*.json'))
print(len(json_list))

# for json_path in json_list:
#     with open(json_path, 'r', encoding='cp949') as f:
#         meta_data = json.load(f)

#     print(meta_data)
#     break

# for json_path in json_list:
#     with open(json_path, 'r', encoding='cp949') as f:
#         meta_data = json.load(f)

#     print(meta_data['description']['impression']['description'])
#     print(meta_data['sketch_info']['img_path'])
#     print(meta_data['org_sketch_info']['img_path'])
#     break


# # target_path = '/home/brad/Dataset/persona-montage/preprocessed'
target_path = '/mnt/hdd1/wkim/DS503/KoDALLE/data/montage/preprocessed'

img_list = []
label_list = []

os.makedirs(target_path, exist_ok=True)
os.makedirs(os.path.join(target_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'labels'), exist_ok=True)

for i, json_path in enumerate(tqdm(json_list)):
    with open(json_path, 'r', encoding='cp949') as f:
        meta_data = json.load(f)

    if 'impression' in meta_data['description']:
        text = meta_data['description']['impression']['description']
    else:
        # print(meta_data['description'][0])
        text = meta_data['description']['face']['description']

    img_path = os.path.join(base_path, meta_data['sketch_info']['img_path'][1:])
    orig_img_path = os.path.join(base_path, meta_data['org_sketch_info']['img_path'][1:])

    target_img_path = os.path.join(target_path, 'images', str(i).zfill(5) + '_' + os.path.splitext(os.path.basename(json_path))[0] + '.jpg')
    target_orig_img_path = os.path.join(target_path, 'images', str(i).zfill(5) + '_' + 'orig_' + os.path.splitext(os.path.basename(json_path))[0] + '.png')

    target_label_path = os.path.join(target_path, 'labels', str(i).zfill(5) + '_' + os.path.splitext(os.path.basename(json_path))[0] + '.txt')
    target_orig_label_path = os.path.join(target_path, 'labels', str(i).zfill(5) + '_' + 'orig_' + os.path.splitext(os.path.basename(json_path))[0] + '.txt')

    img_list.append(target_img_path)
    img_list.append(target_orig_img_path)
    label_list.append(target_label_path)
    label_list.append(target_orig_label_path)

    copy2(img_path, target_img_path)
    copy2(orig_img_path, target_orig_img_path)

    with open(target_label_path, 'w') as f:
        f.write(text)
    with open(target_orig_label_path, 'w') as f:
        f.write(text)

print(len(img_list), len(label_list))

import numpy as np
np.savetxt(os.path.join(target_path, 'image_list.txt'), img_list, fmt='%s')
np.savetxt(os.path.join(target_path, 'label_list.txt'), label_list, fmt='%s')


image_path = Path(os.path.join(target_path))
image_files = [
    *image_path.glob("**/*[0-9].txt"),
    *image_path.glob("**/*[0-9].jpg"),
    *image_path.glob("**/*[0-9].png"),
]
len(image_files) / 2


val_img_list = glob(os.path.join('/mnt/hdd1/wkim/DS503/KoDALLE/data/montage/preprocessed', 'val/*.jpg'))
len(val_img_list)


import numpy as np
np.savetxt(os.path.join('/mnt/hdd1/wkim/DS503/KoDALLE/data/montage/preprocessed', 'image_list_test.txt'), val_img_list, fmt='%s')
