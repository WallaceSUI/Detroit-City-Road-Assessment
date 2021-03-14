import numpy as np
import os
import glob

imgs = glob.glob('/Users/wallace/Desktop/2020Fall/capstone/report-data/boundingbox/*-box-*')
imgs = sorted(imgs, key=lambda name: (name[:-4]))


ff = open('/Users/wallace/Desktop/2020Fall/capstone/report-data/test.txt', 'w')
for i in range(len(imgs)):
    name = imgs[i].split('/')[-1]
    ff.write(name + '\n')

ff.close()