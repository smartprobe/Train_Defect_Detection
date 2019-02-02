import os
import xml.etree.ElementTree as ET

xml_path = '/home/beer/zxf/faster-rcnn.pytorch-master/data/VOCdevkit2007/VOC2007/Annotations'
label_path = '/home/beer/zxf/faster-rcnn.pytorch-master/data/VOCdevkit2007/VOC2007/labels.txt'
train_path = '/home/beer/zxf/faster-rcnn.pytorch-master/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'

lableset = [line.strip() for line in open(label_path)]
train_set = [line.strip() for line in open(train_path)]
error_set = set()
files = os.listdir(xml_path)

for file in files:
    instance = file.split('.')[0]
    if instance not in train_set:
        continue

    file_path = os.path.join(xml_path, file)
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = root.findall('object')

    name_nodes = root.findall('object/name')

    if len(objects) == 0:
        if instance in train_set:
            train_set.remove(instance)
        error_set.add((instance, '0 objects'))

    for name_node in name_nodes:
        if name_node.text not in lableset:
            error_set.add((instance, 'error key ' + name_node.text))
            if instance in train_set:
                train_set.remove(instance)

with open(train_path, 'w') as f:
    for instance in train_set:
        f.write(instance + '\n')

for item in error_set:
    print item
