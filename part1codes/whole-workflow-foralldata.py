import six.moves.urllib as urllib
import os

from xml.etree import ElementTree
from xml.dom import minidom
import collections

import os

import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
#%matplotlib inline

base_path = os.getcwd() + '/RoadDamageDataset/'

damageTypes=["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44"]

# govs corresponds to municipality name.
govs = ["Adachi", "Chiba", "Ichihara", "Muroran", "Nagakute", "Numazu", "Sumida"]

# the number of total images and total labels.
cls_names = []
total_images = 0
for gov in govs:

    file_list = [filename for filename in os.listdir(base_path + gov + '/Annotations/') if not filename.startswith('.')]

    for file in file_list:

        total_images = total_images + 1
        if file == '.DS_Store':
            pass
        else:
            infile_xml = open(base_path + gov + '/Annotations/' + file)
            tree = ElementTree.parse(infile_xml)
            root = tree.getroot()
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                cls_names.append(cls_name)
print("total")
print("# of images：" + str(total_images))
print("# of labels：" + str(len(cls_names)))

# the number of each class labels.
import collections

count_dict = collections.Counter(cls_names)
cls_count = []
for damageType in damageTypes:
    print(str(damageType) + ' : ' + str(count_dict[damageType]))
    cls_count.append(count_dict[damageType])

sns.set_palette("winter", 8)
sns.barplot(damageTypes, cls_count)
plt.show()

# the number of each class labels for each municipality
for gov in govs:
    cls_names = []
    total_images = 0
    file_list = [filename for filename in os.listdir(base_path + gov + '/Annotations/') if not filename.startswith('.')]

    for file in file_list:

        total_images = total_images + 1
        if file == '.DS_Store':
            pass
        else:
            infile_xml = open(base_path + gov + '/Annotations/' + file)
            tree = ElementTree.parse(infile_xml)
            root = tree.getroot()
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                cls_names.append(cls_name)
    print(gov)
    print("# of images：" + str(total_images))
    print("# of labels：" + str(len(cls_names)))

    count_dict = collections.Counter(cls_names)
    cls_count = []
    for damageType in damageTypes:
        print(str(damageType) + ' : ' + str(count_dict[damageType]))
        cls_count.append(count_dict[damageType])

    print('**************************************************')

###############
# Check some images in this dataset
import cv2
import random
import imageio


def draw_images(image_file):
    gov = image_file.split('_')[0]
    img = cv2.imread(base_path + gov + '/JPEGImages/' + image_file.split('.')[0] + '.jpg')

    infile_xml = open(base_path + gov + '/Annotations/' + image_file)
    tree = ElementTree.parse(infile_xml)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymin = int(xmlbox.find('ymin').text)
        ymax = int(xmlbox.find('ymax').text)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # put text
        cv2.putText(img, cls_name, (xmin, ymin - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # draw bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    return img


for damageType in damageTypes:
    tmp = []
    for gov in govs:
        file = open(base_path + gov + '/ImageSets/Main/%s_trainval.txt' % damageType, 'r')

        for line in file:
            line = line.rstrip('\n').split('/')[-1]

            if line.split(' ')[2] == '1':
                tmp.append(line.split(' ')[0] + '.xml')

    random.shuffle(tmp)
    #fig = plt.figure(figsize=(6, 6))
    for number, image in enumerate(tmp[20:21]):
        number = number + 1
        img = draw_images(image)
        #plt.subplot(1, 1, number)
        #plt.axis('off')
        #plt.title('The image including ' + damageType)
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.show()
        imageio.imwrite('training-The image including ' + damageType +'.png', img)

####################
# Next, try road damage detection using SSD_mobilenet!
import numpy as np
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#if tf.__version__ != '1.4.1':
#  raise ImportError('Please upgrade your tensorflow installation to v1.4.1!')

## Object detection imports
#Here are the imports from the object detection module.

from utils import label_map_util

from utils import visualization_utils as vis_util

from imageio import imread
import imageio


# Model preparation
## Variables

#Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  'trainedModels/ssd_mobilenet_innference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'crackLabelMap.pbtxt'

NUM_CLASSES = 8

## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

## Loading label map
#Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Detection
# get images from val.txt
'''
PATH_TO_TEST_IMAGES_DIR = '/home/wallace/Desktop/Detroit-project/RoadDamageDetector-master/RoadDamageDataset/'
D_TYPE = ['D00', 'D01', 'D10', 'D11', 'D20','D40', 'D43']
govs = ['Adachi', 'Ichihara', 'Muroran', 'Chiba', 'Sumida', 'Nagakute', 'Numazu']

val_list = []
for gov in govs:
    file = open(PATH_TO_TEST_IMAGES_DIR + gov + '/ImageSets/Main/val.txt', 'r')
    for line in file:
        line = line.rstrip('\n').split('/')[-1]
        val_list.append(line)
    file.close()

print("# of validation images：" + str(len(val_list)))

TEST_IMAGE_PATHS=[]
random.shuffle(val_list)

for val_image in val_list[0:5]:
    TEST_IMAGE_PATHS.append(PATH_TO_TEST_IMAGES_DIR + val_image.split('_')[0]+ '/JPEGImages/%s.jpg' %val_image)
# Size, in inches, of the output images.
'''
import glob
TEST_IMAGE_PATHS = glob.glob('/Volumes/WALLACE/mapillary-image-015/*')
TEST_IMAGE_PATHS = sorted(TEST_IMAGE_PATHS, key=lambda name: (name[:-4]))


#imgs = glob.glob('data/images/*')
#imgs = sorted(imgs, key=lambda name: int(name[12:-4]))
#imgs1 = imgs[:6]
#imgs2 = imgs[10:]
#imgs = imgs1 + imgs2
masks = glob.glob('/Users/wallace/Desktop/2020Fall/capstone/final/mapillary-pred-015/*')
masks = sorted(masks, key=lambda name: (name[:-4]))

for i in range(len(masks)):
    img = cv2.imread(imgs[i], cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(masks[i], cv2.IMREAD_UNCHANGED) / 255
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
    mask = np.repeat(mask, 3, axis=2)

    finalimg = img * mask

    cv2.imwrite('results/maskresults/maskimg-' + imgs[i].split('/')[-1], finalimg)


IMAGE_SIZE = (12, 8)

import pandas as pd

allinfo = pd.DataFrame(columns=['Image Names', 'Number of Cracks', 'Type D00', 'Type D01', 'Type D10', 'Type D11', 'Type D20', 'Type D40', 'Type D43', 'Type D44', 'Average Detection Score', 'Maximum Detection Score', 'Minimum Detection Score', 'Average Crack Region Percentage', 'Maximum Crack Region Percentage', 'Minimum Crack Region Percentage', 'Average Size of Cracks', 'Maximum Size of Cracks', 'Minimum Size of Cracks'], index=list(np.arange(0, len(TEST_IMAGE_PATHS))))


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for iii in range(len(TEST_IMAGE_PATHS)):
      image_path = TEST_IMAGE_PATHS[iii]
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          allinfo,
          iii,
          image_path,
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          min_score_thresh=0.3,
          use_normalized_coordinates=True,
          line_thickness=8)
      #fig = plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)
      #plt.show()
      imageio.imwrite('results/testing-' + image_path.split('/')[-1], image_np)






