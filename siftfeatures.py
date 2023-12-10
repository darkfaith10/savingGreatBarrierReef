import cv2
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

input0 = '/Users/vijaykumarsingh/Desktop/onemore/'
temp = ['cot', 'not_cot']


for i in temp:
    count = 0
    for filename in os.listdir(input0 + i):
        img = cv2.imread(input0 + i + '/' + filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        sorted_features = sorted(zip(keypoints, descriptors), key=lambda x: x[0].response, reverse=True)

        top_features = np.array([f[1] for f in sorted_features[:300]])

        out = pd.DataFrame(top_features)

        csv_data = out.to_csv('/Users/vijaykumarsingh/Desktop/onemore/SIFT/SIFT_' + i + '.csv', mode='a', index=False)

        count += 1
        if count == 2000:
            break

    print(i + ": " + str(count))


data1 = pd.read_csv('/Users/vijaykumarsingh/Desktop/onemore/SIFT/SIFT_cot.csv', dtype='uint8')
data2 = pd.read_csv('/Users/vijaykumarsingh/Desktop/onemore/SIFT/SIFT_not_cot.csv', dtype='uint8')

data1 = data1.astype('uint8')
data2 = data2.astype('uint8')

combined_data = pd.concat([data1, data2], ignore_index=True)

combined_data.to_csv('/Users/vijaykumarsingh/Desktop/onemore/SIFT/SIFT_combined.csv', index=False)