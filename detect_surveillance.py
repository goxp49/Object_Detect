"""
Created on Tue Jan 16 00:52:02 2018
@author: wang
通过摄像头实时调用API识别画面内容
参考：https://blog.csdn.net/ctwy291314/article/details/80452340
"""

import os
import sys

import cv2
import numpy as np
import tensorflow as tf

# ======================================================================================================================
# 将Tensorflow object detect api目录添加到python搜索范围中
sys.path.append("D:/Anaconda3/Lib/site-packages/tensorflow/models/research/")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# 设置当前项目路径
Project_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_NAME = 'tf_libs/current_export_inference_graph'

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('tf_libs', 'current_export_inference_graph',
                              'label_map.pbtxt')

NUM_CLASSES = 90
# 设置分数阈值
SCORES_THRESHOLD = 0.5
OUTPUT_IMAGE_WIDTH = 768
OUTPUT_IMAGE_HEIGHT = 576

# ======================================================================================================================
# 1.初始化TensorFlow
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 通过TensorFlow API返回目标Roi list
def tf_get_roi(Session, image):
    roi_list = []
    # 获得图片尺寸信息
    image_size = image.shape
    im_width, im_height = image_size[0], image_size[1]
    # print(im_width, im_height)
    # 扩展维度，应为模型期待: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # 每个框代表一个物体被侦测到
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # 每个分值代表侦测到物体的可信度.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # 执行侦测任务，返回的是相对坐标boxes,置信值scores，分类classes，找到目标个数num_detections
    (boxes, scores, classes, num_detections) = Session.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # 将结果转换为二维数组
    boxes = boxes[0]
    classes = classes[0]
    scores = scores[0]
    for i in range(int(num_detections)):
        # 此处依据pbtext中定义选择要标记的分类,本例中person = 1
        if classes[i] == 1 and scores[i] > SCORES_THRESHOLD:
            # 获取每个框的数组（ymin, xmin, ymax, xmax）
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            # 将box坐标转换为图片绝对坐标,如果有自定义输出视频大小则使用自定义大小计算
            if OUTPUT_IMAGE_WIDTH is not None and OUTPUT_IMAGE_HEIGHT is not None:
                (left, right, top, bottom) = (int(xmin * OUTPUT_IMAGE_WIDTH), int(xmax * OUTPUT_IMAGE_WIDTH),
                                              int(ymin * OUTPUT_IMAGE_HEIGHT), int(ymax * OUTPUT_IMAGE_HEIGHT))
            else:
                (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                              int(ymin * im_height), int(ymax * im_height))
            roi_list.append((left, right, top, bottom))
    return roi_list


def show_rectangle(roi_list, image, color=(0, 255, 0)):
    for roi in roi_list:
        (left, right, top, bottom) = roi
        cv2.rectangle(image, (left, top), (right, bottom),
                      color, 2)


def main():
    # camera = cv2.VideoCapture(os.path.join(Project_PATH, 'test_video', 'surveillance.avi'))
    camera = cv2.VideoCapture('test_video/surveillance.avi')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, image = camera.read()
                roi_list = tf_get_roi(sess, image)
                show_rectangle(roi_list, image)

                # cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
                cv2.imshow('object detection', cv2.resize(image, (OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


if __name__ == '__main__':
    main()
