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

# 将Tensorflow object detect api目录添加到python搜索范围中
sys.path.append("D:/Anaconda3/Lib/site-packages/tensorflow/models/research/")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# 设置当前项目路径
Project_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_NAME = 'current_export_inference_graph'

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('current_export_inference_graph',
                              'label_map.pbtxt')

NUM_CLASSES = 90


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


def main():
    # camera = cv2.VideoCapture(os.path.join(Project_PATH, 'test_video', 'surveillance.avi'))
    camera = cv2.VideoCapture('../test_video/surveillance.avi')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, image_np = camera.read()
                # 扩展维度，应为模型期待: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # 每个框代表一个物体被侦测到
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # 每个分值代表侦测到物体的可信度.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # 执行侦测任务.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # 检测结果的可视化
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)

                cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


if __name__ == '__main__':
    main()
