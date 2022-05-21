#coding:utf-8
from __future__ import print_function

import os
import random
import pathlib
import xml.etree.ElementTree as ET

datafolder = r'/project/train/src_repo/yolov5/data/tempdata'
targetfolder = r'/project/train/src_repo/yolov5/data/labels'

def xml_reader(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return width, height, objects


def voc2yolo(filename, targetname):
    classes_dict = {}
    with open("/project/train/src_repo/yolov5/class_name.txt") as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            classes_dict[class_name] = idx
    
    width, height, objects = xml_reader(filename)

    lines = []
    for obj in objects:
        x, y, x2, y2 = obj['bbox']
        class_name = obj['name']
        label = classes_dict[class_name]
        cx = (x2+x)*0.5 / width
        cy = (y2+y)*0.5 / height
        w = (x2-x)*1. / width
        h = (y2-y)*1. / height
        line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
        lines.append(line)

    txt_name = targetname
    with open(txt_name, "w") as f:
        f.writelines(lines)


data_folder = pathlib.Path(datafolder)
target_folder = pathlib.Path(targetfolder)
xml_path_list = list(data_folder.glob('*.xml'))


for path in [path for path in xml_path_list]:
    from_path = str(path)
    to_path = str(os.path.join(target_folder, path.name))
    to_path = to_path.replace('xml', 'txt')
    print(from_path, to_path)

    voc2yolo(from_path, to_path)
    
    
    
    
    
data_path = pathlib.Path('/project/train/src_repo/yolov5/data/images')
all_path_list = data_path.glob("*.jpg")
paths = [str(path) for path in list(all_path_list)]
print(all_path_list)
print(len(paths))
datasize = len(paths)
train_paths = paths[:int(datasize*0.8)]
test_paths = paths[int(datasize*0.8):int(datasize*0.9)]
val_paths = paths[int(datasize*0.9):]
with open('/project/train/src_repo/yolov5/data/train.txt', 'w') as f:
    for p in train_paths:
        f.write(p+'\n')
        
with open('/project/train/src_repo/yolov5/data/test.txt', 'w') as f:
    for p in test_paths:
        f.write(p+'\n')
        
with open('/project/train/src_repo/yolov5/data/val.txt', 'w') as f:
    for p in val_paths:
        f.write(p+'\n')
