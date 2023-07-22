# Interpretable Sparsification of Brain Graphs: Better Practices and Effective Designs for Graph Neural Networks

This code implements the following paper: 

> [Interpretable Sparsification of Brain Graphs: Better Practices and Effective Designs for Graph Neural Networks](https://arxiv.org/abs/2306.14375)


## Requirement
- PyTorch
- PyTorch-Geometric
- numpy

## Running our Method

To train our model, run the following script: 
```shell
python ./src/main.py --method IGS --label_col $TASK_NAME --model $MODEL_NAME
```

where `--label_col` specifies the name of the task you want to work on and `--model` specifies the backbone model you want to use.

## Data

We have provided the data in `./data` directory. To generate your own random data splits,
you could use `process_data_splits` from `./data/data_splits.py` with specified arguments.
The original raw data comes from the [Wu-Minn HCP 1200 subjects data release](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf), and we use pre-processed data from [ConnectomeDB](https://www.humanconnectome.org/software/connectomedb). To learn more about the task information, please visit this [link](https://wiki.humanconnectome.org/display/PublicData/HCP-YA+Data+Dictionary-+Updated+for+the+1200+Subject+Release). 