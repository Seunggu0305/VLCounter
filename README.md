# VLCounter: Text-aware Visual Representation for Zero-Shot Object Counting
## [Paper (ArXiv)]()


![teaser](asset/main.png)

Official Implementation for AAAI 2024 paper VLCounter: Text-aware Visual Representation for Zero-Shot Object Counting


**Update**

🔥🔥🔥 [Dec 9] Our paper is accepted by AAAI 2024.

🔥🔥🔥 [Dec 28] Code and pretrained model is released.

## Contents
* [Preparation](#preparation)
* [Run the Code](#run-the-code)
* [Visualization](#visualization)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)


## Preparation
### 1. Download datasets
In our project, the following datasets are used.
Please visit following links to download datasets:

* [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

* [CARPK](https://lafi.github.io/LPN/)

* [PUCPR+](https://lafi.github.io/LPN/)

* [IOCfish5k](https://github.com/GuoleiSun/Indiscernible-Object-Counting)

### 2. Download required python packages:

The following packages are suitable for NVIDIA GeForce RTX A6000.

```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

If you want to use docker environment, please download the docker image through the command below
```
docker pull sgkang0305/vlcounter
```

### 3. Download CLIP weight and Byte pair encoding (BPE) file

Please download the [CLIP pretrained weight](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) and locate the file under the "pretrain" folder.

Please download the [BPE file](https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz) and locate the file under the "tools/dataset" folder.


## Run the Code

### Train
You can train the model using the following command. Make sure to check the options in the train.sh file.
```
bash scripts/train.sh FSC {gpu_id} {exp_number}
```     


### Evaluation
You can test the performance of trained ckpt with following command. Make sure to check the options in the test.sh file. Especially '--ckpt_used' to specify the specific weight file.
```
bash scripts/test.sh FSC {gpu_id} {exp_number}
```

We provide a [pre-trained ckpt](https://drive.google.com/file/d/1-2lqtsOm9XW4MXhLzrB5Jf9RkXOpDlaQ/view?usp=sharing) of our full model, which has similar quantitative result as presented in the paper. 
| FSC val MAE | FSC val RMSE | FSC test MAE |  FSC test RMSE | 
|-------------|--------------|--------------|----------------|
| 18.06       | 65.13        | 17.05        | 106.16         |

| CARPK MAE | CARPK RMSE | PUCPR+ MAE | PUCPR+ RMSE |
|-----------|------------|------------|-------------|
|  6.46     | 8.68       | 48.94      | 69.08       |


## Visualization
![more](asset/qualitative_vf.png)

## Citation
Consider cite us if you find our paper is useful in your research :).
```

```

## Acknowledgements

This project is based on implementation from [CounTR](https://github.com/Verg-Avesta/CounTR).