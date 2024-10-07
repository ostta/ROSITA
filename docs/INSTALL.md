
### Setup Conda environment

```
# Create a conda environment
conda create -n rosita python=3.8

# Activate the environment
conda activate rosita

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Clone ROSITA code repository 
<!-- and install requirements -->

```
# Clone ROSITA code base
git clone https://github.com/anon-tta/ROSITA

cd ROSITA/
```

<!-- # Install requirements
pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0 -->


### Download Pretrained MaPLe weights and statistics

Download the pretrained MaPLE weights from [here](https://drive.google.com/drive/folders/1Okzw3Ap1y8g6t-Yn0nUpuFFqsFQ7sLzE) and source data statistics of MaPLe from [here](https://drive.google.com/drive/folders/1ls9jWVFzlh-0t_O9dwxbH_IyCRpehQzK) and place it at `ROSITA/weights`. The weights folder should look like

```
weights
    + maple
        - ImgNetpre_vis_means.pt
        - ImgNetpre_vis_vars.pt
        - model.pth.tar-2
```


