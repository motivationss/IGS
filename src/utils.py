import numpy as np
import pandas as pd 
import os 
import torch 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
import os
import random
import numpy as np 
import matplotlib.pyplot as plt

def get_ProcSpeed_AgeAdj():
    root = './data'
    subject_ID = np.loadtxt(os.path.join(root, 'subjectIDs_recon2.txt')).astype(int)
    label = pd.read_csv(os.path.join(root, "HCP_behavioral_data.csv"),index_col="Subject")["ProcSpeed_AgeAdj"]
    label = label.loc[[i for i in subject_ID]]
    label = label.reset_index()["ProcSpeed_AgeAdj"]
    label -= 1
    return label

def process_regression_into_classes(regression_input):
    """
    Turn regression labels into classification labels by the magnitudes
    """
   
    one_third_quantile = np.quantile(regression_input, q=0.333)
    two_third_quantile = np.quantile(regression_input, q=0.666)
    for idx, _ in enumerate(regression_input):
        if regression_input[idx] < one_third_quantile:
            regression_input[idx] = 0 
        elif regression_input[idx] >= one_third_quantile and regression_input[idx] < two_third_quantile:
            regression_input[idx] = 1 
        else:
            regression_input[idx] = 2
    return regression_input 

def edge_info(edge):
    """
    This is a function that print out some basic statisical information of the edge input 
    """
    edge_non_zero = edge[edge!=0]
    print("--------------------------Final Edge Information Including Zeros--------------------------")
    print(f"Mean: {np.mean(edge):3f} ")
    print(f"Median: {np.median(edge):3f}")
    print(f"Max: {np.max(edge):3f}")
    print(f"Standard Deviation : {np.std(edge):3f}")
    print(f"Variance: {np.var(edge):4f}")
    print(f"Range: {np.ptp(edge):4f}")
    print(f"0% Percentile: {np.percentile(edge, 0):4f}")
    print(f"25% Percentile: {np.percentile(edge, 25):4f}")
    print(f"50% Percentile: {np.percentile(edge, 50):4f}")
    print(f"75% Percentile: {np.percentile(edge, 75):4f}")
    print(f"100% Percentile: {np.percentile(edge, 100):4f}")
    print("--------------------------Final Edge Information Including Zeros--------------------------")
    print("\n")
    print("--------------------------Final Edge Information Excluding Zeros--------------------------")
    print(f"Mean: {np.mean(edge_non_zero):3f} ")
    print(f"Median: {np.median(edge_non_zero):3f}")
    print(f"Max: {np.max(edge_non_zero):3f}")
    print(f"Standard Deviation : {np.std(edge_non_zero):3f}")
    print(f"Variance: {np.var(edge_non_zero):4f}")
    print(f"Range: {np.ptp(edge_non_zero):4f}")
    print(f"0% Percentile: {np.percentile(edge_non_zero, 0):4f}")
    print(f"25% Percentile: {np.percentile(edge_non_zero, 25):4f}")
    print(f"50% Percentile: {np.percentile(edge_non_zero, 50):4f}")
    print(f"75% Percentile: {np.percentile(edge_non_zero, 75):4f}")
    print(f"100% Percentile: {np.percentile(edge_non_zero, 100):4f}")
    print("--------------------------Final Edge Information Excluding Zeros--------------------------")

def generate_random_split(indices, nums_train):
    indices = list(indices)
    training_list = list(np.random.permutation(indices)[:nums_train]) 
    test_list = [x for x in indices if x not in training_list]
    return training_list, test_list

def get_subject_indexed_label():
    label_dir = "./data/HCP_behavioral_data.csv"
    label = pd.read_csv(label_dir,index_col="Subject")
    subject_ID = np.loadtxt("./data/subjectIDs_recon2.txt").astype(int)
    label = label.loc[[i for i in subject_ID]]
    label = label.reset_index()
    return label 

def process_regression_label_into_binary_heads_tails(label_col):
    label = get_subject_indexed_label()
    curr_label = label[label_col]
    
    curr_label = pd.Series.to_numpy(curr_label)
    
    higher_quantile = np.nanquantile(curr_label, q=0.66)
    lower_quantile = np.nanquantile(curr_label, q=0.33)
    
    generated_mask = (curr_label <= lower_quantile) | (curr_label >= higher_quantile)    
    label_into_binary = curr_label[generated_mask] 
    label_into_binary_mask = (label_into_binary >= higher_quantile)
    new_labels = label_into_binary_mask * 1 
    
    return generated_mask, new_labels

def get_T2_Count_label():
    root = './data'
    subject_ID = np.loadtxt(os.path.join(root, 'subjectIDs_recon2.txt')).astype(int)
    label = pd.read_csv(os.path.join(root, "HCP_behavioral_data.csv"),index_col="Subject")["T2_Count"]
    label = label.loc[[i for i in subject_ID]]
    label = label.reset_index()["T2_Count"]
    label -= 1
    return label

def get_T1_Count_label():
    root = './data'
    subject_ID = np.loadtxt(os.path.join(root, 'subjectIDs_recon2.txt')).astype(int)
    label = pd.read_csv(os.path.join(root, "HCP_behavioral_data.csv"),index_col="Subject")["T1_Count"]
    label = label.loc[[i for i in subject_ID]]
    label = label.reset_index()["T1_Count"]
    label -= 1
    return label

def matrix_plts(matrix_input, save_address, fig_title):
    plt.imshow(matrix_input)
    plt.colorbar()
    plt.title(fig_title)
    plt.savefig(save_address)

def shuffle(edge, label):
    edge_label_pair_list = []

    for (curr_edge, curr_label) in zip(edge, label):
        edge_label_pair_list.append((curr_edge, curr_label))
    
    edge_label_pair_list = np.random.permutation(edge_label_pair_list)
    shuffled_edge, shuffled_label = zip(*edge_label_pair_list)

    new_edges = np.stack(shuffled_edge)
    new_label = np.stack(shuffled_label)
    
    return new_edges, new_label
    


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_seed(seed):
    """Sets seed"""
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_node(node_dir, subject_ID):
    node = np.loadtxt(os.path.join(node_dir, "100206.txt"))
    node = np.expand_dims(node, axis=0)
    for index in subject_ID[1:]:
        index = int(index)
        curr_node = np.expand_dims(np.loadtxt(os.path.join(node_dir, f"{str(index)}.txt")), axis=0)
        node = np.vstack((node, curr_node))
    node = torch.from_numpy(node)
    torch.save(node, os.path.join(node_dir, "node_features.pt"))

def transform_names(args):
    name1, name2, name3 = "", "", ""

    if args.generate_individual_mask_all:
        name1 = "all"
    elif args.generate_individual_mask_label_specific:
        name1 = "label"
    else:
        raise NotImplementedError
    
    if args.sum_unified_mask:
        name2 = "sum"
    elif args.absolute_sum_unified_mask:
        name2 = "abSum"
    elif args.average_unified_mask:
        name2 = "avg"
    elif args.max_unified_mask:
        name2 = "max"
    elif args.squared_sum_unified_mask:
        name2 = "sqSum"
    
    if args.metaMask_0:
        name3 = 0
    elif args.metaMask_1:
        name3 = 1
    elif args.metaMask_Avg:
        name3 = "avg"
    elif args.metaMask_Sum:
        name3 = "sum"
    elif args.metaMask_AbSum:
        name3 = "abSum"
    elif args.metaMask_Max:
        name3 = "max"
    else:
        raise NotImplementedError
    
    return name1, name2, name3 

from numpy.linalg import eigh
def normalize_edge(edge):
    for idx in range(edge.shape[0]):
        curr_edge = edge[idx, :, :]
        x, v = eigh(curr_edge)
        curr_max_eigen_value = max(x.min(), x.max(), key=abs)
        curr_edge = curr_edge / curr_max_eigen_value
        edge[idx] = curr_edge
    return edge

def zero_gradient(x):
    x.grad.data.zero_()

def iterative_saliency_map(model, edge, x, target):
    edge.requires_grad = True
    out = model(x, edge)
    output_max = out[0, target]
    output_max.backward()
    edge_saliency = torch.squeeze(edge.grad.data.clone())
    zero_gradient(edge)
    return edge_saliency

def prepare_dataset(dimension=100):
    exp_name = './data'
    node_sub_exp_name = 'node_timeseries'
    edge_sub_exp_name = 'netmat'
    subject_name = f"3T_HCP1200_MSMAll_d{str(dimension)}_ts2"
    node_directory = os.path.join(exp_name, node_sub_exp_name)
    edge_directory = os.path.join(exp_name, edge_sub_exp_name)
    node_directory = os.path.join(node_directory, subject_name)
    edge_directory = os.path.join(edge_directory, subject_name)

    edge = np.loadtxt(os.path.join(edge_directory, 'netmats1.txt'))
    edge = np.reshape(edge, (812, dimension, dimension))
    
    subject_ID = np.loadtxt(os.path.join(exp_name, 'subjectIDs_recon2.txt'))
    label = pd.read_csv(os.path.join(exp_name, "HCP_behavioral_data.csv"))
    
    if not os.path.exists(os.path.join(node_directory, "node_features.pt")):
        save_node(node_dir=node_directory, subject_ID=subject_ID)
    node = (os.path.join(node_directory, "node_features.pt"))
    
    G = nx.from_numpy_matrix(edge[0,:,:])
    print(G)
    K = from_networkx(G)
    node = np.loadtxt(os.path.join(node_directory, "100206.txt")).T
    node = torch.from_numpy(node)
    K['x'] = node
    print(K)
    return node, torch.from_numpy(edge), label, torch.from_numpy(subject_ID)