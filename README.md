# Code for Optimal Machine Intelligence at the Edge of Chaos

## Description

This repository is to accompany paper:[Optimal Machine Intelligence at the Edge of Chaos](https://arxiv.org/abs/1909.05176).

## Prerequisites

The Prerequisites for Python are in the Dockerfile. In our group DGX server, use the docker image: dgx/linzhang/tensorflow:20.03-tf2-py3.

## Usage

This repo contains two folders. One for MLP and CNN, and another one is for ResNet and DenseNet.

### Folder1 (Fig.2)

Calculate the Jacobian norm of MLP and CNN in the training process.

(Notation in the code: lyapunov0: method3 in Fig.S1, lyapunov11: method1 in Fig.S1, lyapunov2: method2 in Fig.S1; lyapunov1: jacobian norm.)

Command to run for MLP:

```bash
# For MLP (784-100-784-10)
nohup python main.py --architecture mlp --optimizer Adam --lr 0.001 --epochs 51 --num-iterations 500 --num-repeats 10 &> out/mlp/adam_51_500_10.out &
```

For Fig.2, the MLP's architecture is 784-100-784-10. For Fig.S7(In supplementary), there are different kinds of architecture. To get these results, you need to modify the file models.py and change args.layer = 'dense_2'(line 55 in main.py), which is the number of the last hidden layer. For example, if the architecture is 784-100-100-784-10, the args.layer should be 'dense_3'. (The numbering of the layer is 'dense', 'dense_1', 'dense_2', 'dense_3'.) For different architecture, you can use different totoal running epochs to differ them from the result folder. For example, for 784-100-100-784-10, you can run:

```bash
# For MLP (784-100-100-784-10)
nohup python main.py --architecture mlp --optimizer Adam --lr 0.001 --epochs 52 --num-iterations 500 --num-repeats 10 &> out/mlp/adam_52_500_10.out &
```

After running the above command for 784-100-784-10, two result picture named 'lyapunov1s_21.png' and 'acc_loss_1_21.png' will shown in directory 'results/mlp/Adam_0.9/relu_51_500_10'. They constitute the main part of Fig.2(a). For the Poincare plot in Fig.2(a), you need to check the result picture 'poincare_0.png' and find out the representative epochs for three different phases, then uncomment line 113(main.py) to plot the three poincare plots.

Command to run for CNN:

```bash
# For CNN5
nohup python main.py --architecture cnn --optimizer Adam --lr 0.001 --epochs 51 --num-iterations 500 --num-repeats 10 &> out/cnn/adam_51_500_10.out &
```

Procedure is similar to MLP. Note you need to change the 'figure2a' in line 110-112(main.py) to 'figure2b'.

### Folder2 (Fig.3)

alculate the asymptotic distance of ResNet and DenseNet in the training process.

Command to run for ResNet and DenseNet:

```bash
# For ResNet20
nohup python main.py --architecture resnet --optimizer Adam --lr 0.001 --epochs 1001 --num-iterations 500 --num-repeats 1 &> out/resnet/adam_resnet_1001_500_1.out &
# For DenseNet16
nohup python main.py --architecture densenet --optimizer Adam --lr 0.001 --epochs 201 --num-iterations 500 --num-repeats 1 &> out/densenet/adam_densenet_201_500_1.out &
```

Procedure is similar to MLP. For DenseNet, you need to change 'models_res' as 'models_dense' in line 14(main.py) and change the 'figure3a' in line 159-162(main.py) to 'figure3b'

## Note

This is a simplified version of the code, I did not managed to test all the programs. Thus, please feel free to contact me(linzhang_010@outlook.com) if you have any problems or questions.
