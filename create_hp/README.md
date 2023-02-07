# create_hp (Humanparse)
## Installation 
```shell 
conda create -n [ENV] python=3.7.16
conda activate [ENV]
conda install -c conda-forge cudatoolkit=10.0 cudnn=7.6.5
pip install tensorflow-gpu==1.15
pip install scipy==1.7.3 opencv-python==4.5.5.62 protobuf==3.19.1 Pillow==9.0.1 matplotlib==3.5.1
```

## Run
```shell
# outside of create_hp folder 
python3 -m create_hp
```