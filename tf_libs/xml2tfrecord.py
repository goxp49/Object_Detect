# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 00:52:02 2018
@author: Xiang Guo
将文件夹内所有XML文件的信息记录到CSV文件中
"""
import os
import io
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

import glob
import pandas as pd
import xml.etree.ElementTree as ET

# path = './train_xml'
path = './test_xml'

# 设置当前项目路径
Project_PATH = os.path.dirname(os.path.realpath(__file__))
# XML文件名
TRAIN_CSV_NAME = 'train_labels.csv'
# TRAIN XML文件路径
TRAIN_XML_PATH = os.path.join(Project_PATH, 'train_xml')
# TRAIN CSV文件路径
TRAIN_CSV_PATH = os.path.join(Project_PATH, TRAIN_CSV_NAME)
# TRAIN 图片集路径
TRAIN_IMAGES_DIR_NAME = 'train_images'
# TRAIN TFRecord输出路径
TRAIN_TFRECORD_PATH = os.path.join(Project_PATH, 'train.record')

# TEST文件名
TEST_CSV_NAME = 'test_labels.csv'
# TEST XML文件路径
TEST_XML_PATH = os.path.join(Project_PATH, 'test_xml')
# TEST CSV文件路径
TEST_CSV_PATH = os.path.join(Project_PATH, TEST_CSV_NAME)
# TEST 图片集路径
TEST_IMAGES_DIR_NAME = 'test_images'
# TEST TFRecord输出路径
TEST_TFRECORD_PATH = os.path.join(Project_PATH, 'test.record')


# --------------------------------------------XML TO CSV---------------------------------------------------
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def xml2csv(image_path, csv_name):
    xml_df = xml_to_csv(image_path)
    # xml_df.to_csv('train_labels.csv', index=None)
    xml_df.to_csv(csv_name, index=None)
    print('Successfully converted xml to csv.')


# --------------------------------------------CSV TO XML---------------------------------------------------

# TO-DO replace this with label map
# 此处需要将自己的标签填入！！！！
def class_text_to_int(row_label):
    if row_label == 'daisy':
        return 1
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, images_dir_name):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        print(row['class'])
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def csv2tf(csv_path, tf_path, images_dir_name):
    writer = tf.python_io.TFRecordWriter(tf_path)
    path = os.path.join(os.getcwd(), images_dir_name)
    examples = pd.read_csv(csv_path)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path, images_dir_name)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), tf_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


def main(flag=True, delete=True):
    if flag:
        xml_path = TRAIN_XML_PATH
        csv_name = TRAIN_CSV_NAME
        csv_path = TRAIN_CSV_PATH
        tf_path = TRAIN_TFRECORD_PATH
        images_dir_name = TRAIN_IMAGES_DIR_NAME
    else:
        xml_path = TEST_XML_PATH
        csv_name = TEST_CSV_NAME
        csv_path = TEST_CSV_PATH
        tf_path = TEST_TFRECORD_PATH
        images_dir_name = TEST_IMAGES_DIR_NAME

    xml2csv(xml_path, csv_name)
    csv2tf(csv_path, tf_path, images_dir_name)
    # 是否移除CSV文件
    if os.path.exists(csv_path) and delete:
        os.remove(csv_path)


if __name__ == '__main__':
    main(False)
    main(True)
