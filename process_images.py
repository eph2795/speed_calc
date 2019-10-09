import os
import sys
import pickle

import pandas as pd
import numpy as np
from scipy import ndimage
import matplotlib.image as mpimg

from pyexcel_ods import get_data


if __name__ == '__main__':    
    path = 'soil_results/'
    target_path = 'soil_results/Geom_eval.xls'
    
#     data = get_data(target_path)
#     df = pd.DataFrame(data['Sheet1'][1:], columns=data['Sheet1'][0]).set_index('mod_no')
#     df = pd.DataFrame(data['Sheet1'][1:], columns=data['Sheet1'][0]).set_index("'mod_no'")
    df = pd.read_excel(target_path, header=0).set_index("mod_no")
    images = []
    Y = []
    for i, fname in enumerate(sorted(os.listdir(path))):
        if len(fname.split('.')) != 2:
            continue
        name, ext = fname.split('.')
        if ext != 'bmp':
            continue
#         Y.append(df.loc[name, 'cd/area^2'])
        try:
#             y = float(df.loc["'" + name + "'", "'cd/area^2'"][1:-1])
            y = float(df.loc[name, "'cd/area^2'"][1:-1])
            Y.append(y)
        except KeyError:
            print('No results in table: {}!'.format(name))
            
        fpath = os.path.join(path, fname)
        img = mpimg.imread(fpath)
        if len(img.shape) > 2:
            img = img[:, :, 0]
            
        img = (255 - img).astype(bool)
        x, y = np.where(img)
        cropped = img[np.min(x) : np.max(x)+1, np.min(y) : np.max(y)+1]
        size = max(*cropped.shape)
        padded = np.zeros((int(size * 1.5) + 1, int(size * 1.5) + 1))
        l = int(0.25 * size) + 1
        u = int(0.25 * size) + 1
        padded[l : l+cropped.shape[0], u : u+cropped.shape[1]] = cropped
        img = np.repeat(padded.reshape(padded.shape[0], padded.shape[1], 1), 3, axis=2).astype(np.float64)
        images.append(img)
    Y = np.array(Y, dtype=np.float64)
    print(len(Y))
    with open(os.path.join(os.path.split(path)[0], 'soil_images.pkl'), 'wb') as f:
        pickle.dump(obj=(images, Y), file=f)