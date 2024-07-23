# PRET: Planning with Directed Fidelity Trajectory for Vision and Language Navigation

## Requirements
1. Install [Matterport3D simulator](https://github.com/peteanderson80/Matterport3DSimulator).

    Put `connectivity` in `./`(or create a soft link).
    Put `build` and Matterport3D data `v1` in the `./data` direcotry(or create a soft link).

2. Install python requirements.

    ```text
    timm==0.9.5
    transformers==4.28.1
    torch==1.13.0, do not use >=2.0

    numpy
    pandas
    matplotlib
    python-opencv

    tqdm
    pyyaml
    networkx
    jsonlines
    ```

3. Download datasets, image features and model checkpoints from [here](https://drive.google.com/drive/folders/1U6MoboyOaDtm2dYiMr1mR4dtNYxahNEH?usp=sharing).
    Download the `data.zip` and `log.zip` and unzip them.


Finally, the directory layout should looks like:
```text
.
├── connectivity
├── data
│   ├── build
│   ├── candidate_buffer.pkl
│   ├── img_features
│   ├── pretrain_data
│   ├── pretrained_model
│   ├── R2R
│   ├── RxR
│   └── v1
├── log
│   └── commit
├── pretrain.sh
├── README.md
├── run.sh
└── src
```

## Train
Pretrain:
```bash
sh pretrain.sh R2R
# sh pretrain.sh RxR
```
Note that even though multi-process training is implemented, I never use it. Therefore, there may be some bugs.

Fine-tune:
```bash
sh run.sh R2R
# sh run.sh RxR
```
