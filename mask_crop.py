import cv2
from skimage import io
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

path='input/train_masks'
f2=open('train.pkl','rb')
load_list=pickle.load(f2)
f2.close()

for img_ids,value in load_list.items():
    filename=os.path.join(path,'{}'.format(img_ids)+'_mask.jpg')
    img_orig=io.imread(filename)
    x1,y1,x2,y2=value[0],value[1],value[2],value[3]
    img_mask=img_orig[y1  : y2 , x1 : x2]
    #plt.imshow(img_mask,cmap='gray')
    #plt.axis('off')
    io.imsave(os.path.join('train_mask_crop/', '{}'.format(img_ids)+'_mask.jpg'),img_mask)
    #plt.savefig(os.path.join('train_mask_crop/', '{}'.format(img_ids)+'_mask.jpg'))


