Unpriortized Autoencoder: PyTorch Implementation
===========================================================

This repository is the PyTorch implementation of  
"[Unpriortized Autoencoder For Image Generation](https://ieeexplore.ieee.org/document/9191173)"

Environment
-----------
- PyTorch (0.4>=)

Directory Structure
-------------------
```
(root-directory)
├── README.md
├── run_mmod.py
├── src
│   └── (python-source-file.py)
├── result
│   └── (result-directory)
└── data
    └── coco-2017
        ├── annotations
        └── images
```

Usage
-----
Training
```
# bash train_ae.sh
# base train_armdn.sh
```

Citation
--------
```
@inproceedings{yoo2020unpriortized,
  title={Unpriortized Autoencoder For Image Generation},
  author={Yoo, Jaeyoung and Lee, Hojun and Kwak, Nojun},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  pages={763--767},
  year={2020},
  organization={IEEE}
}
```


