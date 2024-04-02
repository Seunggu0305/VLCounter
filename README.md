# VLCounter: Text-aware Visual Representation for Zero-Shot Object Counting
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vlcounter-text-aware-visual-representation/zero-shot-counting-on-fsc147)](https://paperswithcode.com/sota/zero-shot-counting-on-fsc147?p=vlcounter-text-aware-visual-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vlcounter-text-aware-visual-representation/object-counting-on-carpk)](https://paperswithcode.com/sota/object-counting-on-carpk?p=vlcounter-text-aware-visual-representation)
## [Paper (ArXiv)](https://arxiv.org/abs/2312.16580)


![teaser](asset/main.png)

Official Implementation for AAAI 2024 paper VLCounter: Text-aware Visual Representation for Zero-Shot Object Counting


**Update**

🔥🔥🔥 [Dec 9] Our paper is accepted by AAAI 2024.

🔥🔥🔥 [Dec 28] Code and pretrained model are released.

## Contents
* [Preparation](#preparation)
* [Run the Code](#run-the-code)
* [Visualization](#visualization)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)


## Preparation
### 1. Download datasets
In our project, the following datasets are used.
Please visit the following links to download datasets:

* [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

* [CARPK](https://lafi.github.io/LPN/)

* [PUCPR+](https://lafi.github.io/LPN/)

* [IOCfish5k](https://github.com/GuoleiSun/Indiscernible-Object-Counting)
  
We use CARPK and PUCPR+ by importing the hub package. Please click [here](https://datasets.activeloop.ai/docs/ml/datasets/carpk-dataset/) for more information.
```
/
├─VLCounter/
│
├─FSC147/    
│  ├─gt/
│  ├─image/
│  ├─ImageClasses_FSC147.txt
│  ├─Train_Test_Val_FSC_147.json
│  ├─annotation_FSC147_384.json
│  
├─IOCfish5k/
│  ├─annotations/
│  ├─images/
│  ├─test_id.txt/
│  ├─train_id.txt/
│  ├─val_id.txt/
```


### 2. Download required Python packages:

The following packages are suitable for NVIDIA GeForce RTX A6000.

```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install hub
```

If you want to use the docker environment, please download the docker image through the command below
```
docker pull sgkang0305/vlcounter
```

### 3. Download CLIP weight and Byte pair encoding (BPE) file

Please download the [CLIP pretrained weight](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) and locate the file under the "pretrain" folder.

Please download the [BPE file](https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz) and locate the file under the "tools/dataset" folder.


## Run the Code

### Train
You can train the model using the following command. Make sure to check the options on the train.sh file.
```
bash scripts/train.sh FSC {gpu_id} {exp_number}
```     


### Evaluation
You can test the performance of trained ckpt with the following command. Make sure to check the options in the test.sh file. Especially '--ckpt_used' to specify the specific weight file.
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
Consider citing us if you find our paper useful in your research :).
```
@inproceedings{kang2024vlcounter,
  title={VLCounter: Text-Aware Visual Representation for Zero-Shot Object Counting},
  author={Kang, Seunggu and Moon, WonJun and Kim, Euiyeon and Heo, Jae-Pil},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={3},
  pages={2714--2722},
  year={2024}
}
```

## Acknowledgements

This project is based on implementation from [CounTR](https://github.com/Verg-Avesta/CounTR).
