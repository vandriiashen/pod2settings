# POD2Settings

This repository contains the code used in the paper "X-ray Image Generation As A Method Of Performance Prediction For Real-Time Inspection: A Case Study".
It can be used for data generation, training and testing DCNNs, and performance evaluation using POD curves.

## Script Description

**calibration.py** was used to extract noise model parameters from the series of flatfield images.

**real_data_process.py** was used to make experimental datasets from raw data. The scipt also contains the code for computing dual-energy contrast used in POD curves.

**noise_test.py** was used to generate noisy datasets corresponding to different values of exposure time.

**chichen.py** was used to train and test DCNNs.

**visualize_pod.py** was used to plot POD curves, compare performance on real and generated data, and connect the performance estimate with exposure time.
