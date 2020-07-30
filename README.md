# esekf_fusion

This repository is getting updated everyday...

### Code

-`ms_ekf.py`      - Will implement Multi-Rate Kalman Filter.
-`esekf_jsola.py` - Implementation of esekf based on J. Sola's work(not tested).

### Datasets

#### Kitti
- Kitti dataset used: 2011_09_26_drive_0001 http://www.cvlibs.net/datasets/kitti/raw_data.php
- From these data I generated: `imu.txt`, `gt.txt` and `vodataset.txt`.
- Visual pose is generated using ORB-SLAM2

#### TUM MH_01_easy dataset

- Inside folder data/imu0/ you can find:
-`data.csv` which contains imu data
-`f_mono.txt` which contains visual pose generated using new ORB-SLAM3
-`ground_truth.csv` contains ground truth.