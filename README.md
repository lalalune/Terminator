![Python >=3.9](https://img.shields.io/badge/Python->=3.8-blue.svg)
![PyTorch >=1.11](https://img.shields.io/badge/PyTorch->=1.1-yellow.svg)


# Terminator (unfinished & rechecking...)
 
HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction (https://arxiv.org/pdf/2401.17948.pdf)



## Updates

1. I try to open source all the code today.


### Overall Architecture
- [x] No residual connections
- [x] A new variant of self-attention and FWP to get fast weights - `HyperZ⋅Z⋅W` operator
- [x] Defination of `Full-context Interaction`
- [x] A multi-branch structure - `Slow-Fast Neural Encoding (SFNE)` Block
- [x] A local `slow neural loss` to obtain accurate pixel-level scores (i.e. pixel-level "attention")
- [x] Replace Normlization with `Standardization`: no Affine parameters and Momentum argument
- [x] `Four new components`: G-IBS, RGU, Hyper-Channel Interaction, Hyper Interaction
- [x] `Excellent Properties`: faster training convergence, zero-mean features, fewer model parameters
- [x] New optimizers `Kar2` & `Kar3` are more suitable for slow-fast architectue
- [x] The width of NN is far greater than its depth (`5000 vs. 5`) 


### Installation

We provide an environment file ``environment.yml`` containing the required dependencies. Clone the repo and run the following command in the root of this directory:
```
conda env create -f environment.yml
```

### Expriments

#### Training commands
```shell
python main.py dataset.name="CIFAR100" optimizer.name="Kar3" optimizer.lr=0.00120925 optimizer.l2_reg=0.00004 optimizers.betas=[0.75, 0.99]
```

#### Testing commands
```shell
python main.py dataset.name="CIFAR100" train.do=False pretrained.load=True pretrained.filename='**'
```

If you want distrubuted training, please add: ```train.distributed=True, num_nodes=1, avail_gpus=4```


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
Thank [Alex Yanko](https://x.com/LeopolisDream) for posting this work on X ([twitter](https://x.com/LeopolisDream/status/1804627325583327358)), which attracts more attention. Harvie Zhang independently conducted this project, which commenced in October 2022 and received 1/4 of its funding from Innovation and Technology Fund. Our code is learned from [ccnn](https://github.com/david-knigge/ccnn). We hope our `Terminator` could promote the exploration of novel network architecture.


## Dicussion

If you have any suggestion or question, you can contact me by harviezzw@gmail.com. Thanks for your attention!
