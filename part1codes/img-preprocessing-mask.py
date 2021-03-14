import numpy as np
import glob
import cv2
'''
imgs = glob.glob('data/images/*')
imgs = sorted(imgs, key=lambda name: int(name[12:-4]))
imgs1 = imgs[:6]
imgs2 = imgs[10:]
imgs = imgs1 + imgs2
masks = glob.glob('data/labels/*_pixels0.png')
masks = sorted(masks, key=lambda name: int(name[12:-12]))

for i in range(49):
    img = cv2.imread(imgs[i], cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(masks[i], cv2.IMREAD_UNCHANGED) / 255
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
    mask = np.repeat(mask, 3, axis=2)

    finalimg = img * mask

    cv2.imwrite('results/maskresults/maskimg-' + imgs[i].split('/')[-1], finalimg)
'''



'''
im = cv2.imread('results/mask-detection/testing-maskimg-10.jpg')
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
idx =0
for cnt in contours:
    idx += 1
    x,y,w,h = cv2.boundingRect(cnt)
    roi=im[y:y+h,x:x+w]
    cv2.imwrite(str(idx) + '.jpg', roi)
    #cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
cv2.imshow('img',im)
cv2.waitKey(0)
'''

import glob
import os


imgs = glob.glob('/Users/wallace/Desktop/2020Fall/capstone/report-data/200data_changeview/*')       ##change the path for images after views
imgs = sorted(imgs, key=lambda name: (name[:-4]))

masks = glob.glob('/Users/wallace/Desktop/2020Fall/capstone/report-data/200datamask/*')             ##change the path for masks
masks = sorted(masks, key=lambda name: (name[:-4]))

file_name = '/Users/wallace/Desktop/2020Fall/capstone/report-data/200data_changeview_addmask'       ##change the path for saving images
if not os.path.exists(file_name):
    os.makedirs(file_name)

for i in range(len(masks)):
    if imgs[i].split('/')[-1] != masks[i].split('/')[-1]:
        print('wrong match')
    img = cv2.imread(imgs[i], cv2.IMREAD_UNCHANGED)
    print(imgs[i])
    mask = cv2.imread(masks[i], cv2.IMREAD_UNCHANGED) / 255
    print(masks[i])
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
    mask = np.repeat(mask, 3, axis=2)

    finalimg = img * mask

    cv2.imwrite(file_name + '/' + imgs[i].split('/')[-1], finalimg)
    #print('finish num: ', i, ' with name: ', imgs[i].split('/')[-1])