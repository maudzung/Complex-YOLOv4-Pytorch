"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Refer: https://github.com/ghimiredhikura/Complex-YOLOv3
"""

import os

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dataset_dir = '../../dataset/kitti'
    train_file = open(os.path.join(dataset_dir, 'train.txt'), 'w')
    val_file = open(os.path.join(dataset_dir, 'val.txt'), 'w')
    file_ids = ["%06d\n" % i for i in range(0, 7481)]
    train_ids, val_ids = train_test_split(file_ids, test_size=0.5037)
    print(train_ids)
    train_file.writelines(train_ids)
    val_file.writelines(val_ids)
    train_file.close()
    val_file.close()
