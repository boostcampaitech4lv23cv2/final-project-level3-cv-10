# Backend code for communication between main server and preprocessing servers

## Preparation
```shell
# clone github 
git clone https://github.com/boostcampaitech4lv23cv2/final-project-level3-cv-10.git Final_Project
cd Final_Project
git checkout features/connect
```

## create_hp (Humanparse)
### Installation 
```shell 
conda create -n [ENV] python=3.7.16
conda activate [ENV]
conda install -c conda-forge cudatoolkit=10.0 cudnn=7.6.5
pip install tensorflow-gpu==1.15
pip install scipy==1.7.3 opencv-python==4.5.5.62 protobuf==3.19.1 Pillow==9.0.1 matplotlib==3.5.1
```

### Run
```shell
# outside of create_hp folder 
python3 -m create_hp
```

## create_mask 
### Installation
```shell
conda create -n [ENV] python=3.10
conda activate [ENV]
pip install carvekit --extra-index-url https://download.pytorch.org/whl/cu110
```
### Run 
```shell
# outside of create_mask folder
python3 -m create_mask
```


## densepose 