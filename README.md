# Complexer_YOLO

## Data Preparation
Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

The downloaded data includes:

- Velodyne point clouds **(29 GB)**: input data to VoxelNet
- Training labels of object data set **(5 MB)**: input label to VoxelNet
- Camera calibration matrices of object data set **(16 MB)**: for visualization of predictions
- Left color images of object data set **(12 GB)**: for visualization of predictions

## Source code structure

```
├── README.md    
└── dataset/    
    └──kitti
        ├── training
        |   ├── image_2 <-- for visualization
        |   ├── calib
        |   ├── label_2
        |   ├── velodyne
        └── testing  
        |   ├── image_2 <-- for visualization
        |   ├── calib
        |   ├── velodyne 
└──src/
```