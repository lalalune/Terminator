![Python >=3.9](https://img.shields.io/badge/Python->=3.8-blue.svg)
![PyTorch >=1.11](https://img.shields.io/badge/PyTorch->=1.1-yellow.svg)


# Terminator (unfinished & rechecking...)
 
HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction (https://arxiv.org/pdf/2401.17948.pdf)



## Updates

1. I try to open source the code in the next two days.

2. I seem to have got better results than before!


### Installation

We provide an environment file; ``environment.yml`` containing the required dependencies. Clone the repo and run the following command in the root of this directory:
```
conda env create -f environment.yml
```

### Train

#### Training commands
```shell
python main.py dataset.name="CIFAR10" optimizer.name="Kar3" optimizer.lr=0.001215 optimizer.l2_reg=0.00005 optimizers.betas=[0.5, 0.95]
```

#### Testing commands
```shell
python main.py dataset.name="CIFAR10" train.do=False pretrained.load=True pretrained.filename='**'
```


## Citation
If you use this codebase, or otherwise find our work valuable, please cite our paper:
```
@article{zhang2024hyperz,
  title={HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction},
  author={Zhang, Harvie},
  journal={arXiv preprint arXiv:2401.17948},
  year={2024}
}
```


## Acknowledgement
Thank Alex Yanko for posting this work on X ([twitter](https://x.com/LeopolisDream/status/1804627325583327358)), which attracts more attention. Harvie Zhang independently conducted this project, which commenced in October 2022 and received 1/4 of its funding from Innovation and Technology Fund. Our code is learned from [ccnn](https://github.com/david-knigge/ccnn). We hope our `Terminator` could promote the exploration of novel network architecture.


## Dicussion

If you have any suggestion or question, you can contact me by harviezzw@gmail.com. Thanks for your attention!
