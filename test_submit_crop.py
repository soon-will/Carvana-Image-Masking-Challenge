import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.u_net import get_unet_512,get_unet_128
import pickle
from skimage import io,transform
import matplotlib.pyplot as plt

input_size = 512
batch_size = 1
threshold = 0.5
model = get_unet_512()


df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

#path='input/train_masks'
f2=open('test.pkl','rb')
dicts=pickle.load(f2)
f2.close()

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


rles = []

model.load_weights(filepath='weights/best_weights.hdf5')

print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
for start in tqdm(range(0, len(ids_test), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch.values:
        img = io.imread('input/test_crop/{}.jpg'.format(id))
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)

    img_orig = np.zeros((batch_size,1280,1918))
    for i, id in enumerate(ids_test_batch):
        orig_height,orig_width=(io.imread('input/test_crop/{}.jpg'.format(id))).shape[0],(io.imread('input/test_crop/{}.jpg'.format(id))).shape[1]
        #pred=io.imread('input/test_crop/{}.jpg'.format(id))
        #print(preds[i].shape)
        #print(orig_width,orig_height)
        x1,y1,x2,y2 = dicts[id]

        #print(x1,y1,x2,y2)
        temp = cv2.resize(preds[i], (orig_width, orig_height))
        #print(temp.shape)
        img_orig[i,y1: y2,x1: x2]= temp

        #io.imsave('test_mask_{}.jpg'.format(id),img_orig[i])
        '''plt.imshow(img_orig[i], cmap='gray')
        plt.axis('off')
        plt.show()'''

    for pred in img_orig:
        mask = pred > threshold
        rle = run_length_encode(mask)
        rles.append(rle)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/crop_submission.csv.gz', index=False, compression='gzip')
