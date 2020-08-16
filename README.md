# MP Project: Eye Gaze Estimation

## Requirements
The following python libraries should be installed:

```
tensorflow-gpu==1.13.0 or 1.12.0
coloredlogs
h5py
numpy
opencv-python
ujson
```

## How to run our Code
Start by checking out this git repo.

If you want to run the code on leonhard, just run the following commands:

```
cd code_tf1_13/src

python train_gaga.py
```

If you want to run the code on your personal machine, you need to change the location of the datasets first. Open the `data_location.py` file, which can be found in the `/code_tf1_13/src/configs` folder. And then change the following line:

```
9:        dataconfig['location'] = $PATH TO DATASETS$
```

All parameters are set to what we used for our submitted solution.



## Some used References
[Full-Face Appearance-Based Gaze Estimation](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w41/papers/Bulling_Its_Written_All_CVPR_2017_paper.pdf)

[MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation](https://arxiv.org/pdf/1711.09017.pdf)

[ResNet-34](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

[DenseNet](https://arxiv.org/pdf/1608.06993.pdf)

