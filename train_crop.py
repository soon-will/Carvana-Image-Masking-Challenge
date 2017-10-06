import numpy as np
import cv2
from skimage.measure import compare_ssim
import os
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import pickle



path = 'input/train/'
df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('_')[0])
ids_train=list(set(ids_train))

bboxes = {}

for i in range(0,len(ids_train)):
    img_id = ids_train[i]


    # You need to input the path (train or test) and the car_id (img_id)

    # Let's iterate over all the angles
    for num in range(1, 17):

        # Here we read images i and i+1.
        # If i==16, we will read the first image
        # To speed up the things, we can scale the images 5 times
        fname1 = os.path.join(path, '{}'.format(img_id) + '_{:0>2}.jpg'.format(num))
        fname2 = os.path.join(path, '{}'.format(img_id)+ '_{:0>2}.jpg'.format((num) % 16 + 1))
        img_1_orig = io.imread(fname1)
        h, w = img_1_orig.shape[0], img_1_orig.shape[1],
        img_1_scaled = cv2.resize(img_1_orig, (w // 5, h // 5))

        img_2_orig = io.imread(fname2)
        h, w = img_2_orig.shape[0], img_2_orig.shape[1],
        img_2_scaled = cv2.resize(img_2_orig, (w // 5, h // 5))



        # As the images differ from each other just a by a small angle of rotation,
        # We can find their difference and draw a boundign box around the obtained image
        img1 = cv2.cvtColor(img_1_scaled, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img_2_scaled, cv2.COLOR_BGR2GRAY)

        # Instead of plain difference, we look for structural similarity
        score, dimg = compare_ssim(img1, img2, full=True)
        dimg = (dimg * 255).astype("uint8")

        plt.figure()
        plt.imshow(dimg)
        plt.show()

        thresh = cv2.threshold(dimg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ROIS = []
        for c in cnts[1]:
            (x, y, w, h) = cv2.boundingRect(c)
            # We dont want to use too small bounding boxes
            if w * h > img1.shape[0] * img1.shape[1] // 9:
                ROIS.append([x, y, x + w, y + h])

        ROIS = np.array(ROIS)

        # Now we will draw a boundig box
        # around all the bounding boxes (there are outliers)
        x1 = ROIS[:, 0].min()
        y1 = ROIS[:, 1].min()

        x2 = ROIS[:, 2].max()
        y2 = ROIS[:, 3].max()

        #bboxes.append([fname1, x1 * 5, y1 * 5, x2 * 5, y2 * 5])


        bboxes['{}'.format(img_id)+'_{:0>2}'.format(num)]=[x1 * 5, y1 * 5, x2 * 5, y2 * 5]

        #print(bboxes)

        



        #plt.figure()
        img1=img_1_orig[y1 * 5 : y2 * 5, x1 * 5: x2*5,: ]
        #plt.imshow(img1)
        #plt.axis('off')
        #io.imsave(os.path.join('train_crop/','{}'.format(img_id)+'_{:0>2}'.format(num)+'.jpg'),img1)

f =open('train.pkl', 'wb')
pickle.dump(bboxes, f)
f.close()
