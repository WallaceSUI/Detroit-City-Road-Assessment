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
cls_count[0] = 0
cls_count[1] = 8
cls_count[2] = 0
cls_count[3] = 0
cls_count[4] = 1
cls_count[5] = 0
cls_count[6] = 0
cls_count[7] = 27
sns.barplot(damageTypes, cls_count)
plt.show()


print('total')
print('# of images : 53')
print('# of labels detected : 36')
print('D00: 0')
print('D01: 8')
print('D10: 0')
print('D11: 0')
print('D20: 1')
print('D40: 0')
print('D43: 0')
print('D44: 27')