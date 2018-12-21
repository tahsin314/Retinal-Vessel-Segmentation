# Retinal Vessel Segmentation
This repository contains codes for the *autoencoder* from the paper [Retinal Vein detection using Residual Block Incorporated U-Net Architecture and Fuzzy Inference System
](https://www.researchgate.net/publication/329164442_Retinal_Vein_detection_using_Residual_Block_Incorporated_U-Net_Architecture_and_Fuzzy_Inference_System). I used the train set of [*DRIVE*](https://www.isi.uu.nl/Research/Databases/DRIVE/) dataset for training and
it's test dataset along with data from [*STARE*](http://cecas.clemson.edu/~ahoover/stare/) for testing.

Here are the performance metrics measured on both datasets using my model.

|   Metric    |         Drive         |         Stare          |
| :---------: | :-------------------: | :--------------------: |
|  Accuracy   | First manual: 0.9675  |   A. Hoover: 0.9537    |
|             | Second manual: 0.9712 | V. Kouznetsova: 0.9314 |
|  Precision  | First manual: 0.8453  |   A. Hoover: 0.7689    |
|             | Second manual: 0.8486 | V. Kouznetsova: 0.8622 |
| Sensitivity | First manual: 0.7690  |   A. Hoover: 0.5582    |
|             | Second manual: 0.8013 | V. Kouznetsova: 0.4380 |
| Specificity | First manual: 0.9865  |   A. Hoover: 0.9862    |
|             | Second manual: 0.9869 | V. Kouznetsova: 0.9915 |
|     NPV     | First manual: 0.9781  |   A. Hoover: 0.9645    |
|             | Second manual: 0.9781 | V. Kouznetsova: 0.9645 |
|     AUC     | First manual: 0.9818  |   A. Hoover: 0.9223    |
|             | Second manual: 0.9848 | V. Kouznetsova: 0.8987 |

### Requirements
* Numpy
* OpenCV
* Keras
* Tensorflow
* Scikit-learn

### Dataset Preprocessing
All images were manually converted to **.tiff* since *OpenCV* can not read images from *DRIVE* and *STARE* with other formats.
The images were extracted and kept under the directory `./data/`. 

### Training
Run `train.py`. Change label name with your labels in line `28` and `42`

### Evaluation
Run `eval.py`. Change label name with your lables in line `21`