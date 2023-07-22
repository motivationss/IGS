import numpy as np 
import random 
import pandas as pd 
import os 

meaingfulcol_name_list = ["PicSeq_AgeAdj", "CardSort_AgeAdj", "Flanker_AgeAdj",
"ReadEng_AgeAdj", "PicVocab_AgeAdj", "ListSort_AgeAdj"]

def get_subject_indexed_label():
    address = "../"
    label_dir = f"{address}data/HCP_behavioral_data.csv"
    label = pd.read_csv(label_dir,index_col="Subject")
    subject_ID = np.loadtxt(f"{address}data/subjectIDs_recon2.txt").astype(int)
    label = label.loc[[i for i in subject_ID]]
    label = label.reset_index()
    return label  


def count_distribution_label(label, attribute='full'):
    ones = np.sum(label)
    total = len(label)
    zeros = total - ones 
    print(f"{attribute}:" )
    print("ones: ", ones)
    print("zeros: ", zeros)


def process_equal_labels(converted_label):
    """ Ensure the processed binary classification is balanced """
    ones = np.sum(converted_label)
    total = len(converted_label)
    zeros = total - ones 

    generated_mask = np.array([True for _ in range(total)])

    if ones > zeros:
        index_ones = np.where(converted_label==1)
        index_ones = [t for t in index_ones[0]]
        removed_index_ones = index_ones[zeros:]
        generated_mask[removed_index_ones] = False 
    elif zeros > ones:
        index_zeros = np.where(converted_label==0)
        # Squeeze the tuple 
        index_zeros = [t for t in index_zeros[0]]
        removed_index_zeros = index_zeros[ones:]
        
        generated_mask[removed_index_zeros] = False  

    label = converted_label[generated_mask]
    
    return generated_mask, label 

epils = 10-5
def process_regression_label_into_binary_heads_tails(label_col):
    label = get_subject_indexed_label()
    curr_label = label[label_col]
    curr_label.fillna(curr_label.median(skipna=True), inplace=True)
    # To make sure the Nan value does not interfere with the head/tail split
    curr_label = pd.Series.to_numpy(curr_label)
    
    
    higher_quantile = np.quantile(curr_label, q=0.666666666666666666666666666666666)
    lower_quantile = np.quantile(curr_label, q=0.33333333333333333333333333333333)
    # print("higher quantile: ", higher_quantile)
    # print("lower quantile: ", lower_quantile)
    # print("number of higher: ", np.sum( curr_label >= higher_quantile))
    # print("number of lower: ", np.sum(curr_label <= lower_quantile))
    
    generated_mask = (curr_label <= lower_quantile) | (curr_label >= higher_quantile)
    
    label_into_binary = curr_label[generated_mask] 
    
    label_into_binary_mask = (label_into_binary >= higher_quantile)
    
    new_labels = label_into_binary_mask * 1 
    mask_two, balanced_label = process_equal_labels(new_labels)

    return generated_mask, mask_two,  balanced_label

from numpy.linalg import eigh
def normalize_reshaped_edge(edge):
    for idx in range(edge.shape[0]):
        curr_edge = edge[idx, :, :]
        x, v = eigh(curr_edge)
        curr_max_eigen_value = max(x.min(), x.max(), key=abs)
        curr_edge = curr_edge / curr_max_eigen_value
        edge[idx] = curr_edge
    return edge

def load_brain_data(dimension=100, label_col="PicVocab_AgeAdj", normalize=False):
    root = './data'
    
    edge_sub_exp_name = 'netmat'
    subject_name = f"3T_HCP1200_MSMAll_d{str(dimension)}_ts2"
    edge_directory = os.path.join(root, edge_sub_exp_name)
    edge_dir = os.path.join(edge_directory, subject_name)
    
    edge = np.loadtxt(os.path.join(edge_dir, 'netmats1.txt'))
    edge[edge>3.8666] = 0
    edge[edge<-4.020201] = 0 
    edge = np.reshape(edge, (812, dimension, dimension))

    # [Optional] Normalize it directly before feeding into dataloader/otherbaselines
    if normalize:
        edge = normalize_reshaped_edge(edge)

    subject_ID = np.loadtxt(os.path.join(root, 'subjectIDs_recon2.txt')).astype(int)

    label = pd.read_csv(os.path.join(root, "HCP_behavioral_data.csv"),index_col="Subject")[label_col]
    label = label.loc[[i for i in subject_ID]]
    label = label.reset_index()[label_col]

    print(f"You are using {label_col} as the label!")
    if label_col == "T2_Count":
        label -= 1 
        label = label.astype(int)
    elif label_col == "T1_Count":
        label -= 1 
        label = label.astype(int)
    elif label_col in meaingfulcol_name_list:
        generated_mask, mask_two, label = process_regression_label_into_binary_heads_tails(label_col)
        edge = edge[generated_mask][mask_two]
    else:
        raise NotImplementedError


    return edge, label 


def train_val_test_split(edge, label):
    order = np.random.permutation(edge.shape[0]) 

    shuffled_edge = edge[order] 
    shuffled_label = label[order]

    train_split = int(edge.shape[0] * 0.7)
    val_split = int(edge.shape[0] * 0.85)
    
    train_edge = shuffled_edge[:train_split]
    train_label = shuffled_label[:train_split] 

    val_edge = shuffled_edge[train_split:val_split]
    val_label = shuffled_label[train_split:val_split]

    test_edge = shuffled_edge[val_split:]
    test_label = shuffled_label[val_split:]

    return train_edge, train_label, val_edge, val_label, test_edge, test_label 


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def make_equal_split(train_edge, val_edge, test_edge, train_label, val_label, test_label):
    num_ones_trains = np.sum(train_label)
    num_zeros_trains = len(train_label) - num_ones_trains 

    num_ones_vals = np.sum(val_label)
    num_zeros_vals = len(val_label) - num_ones_vals 

    num_ones_tests = np.sum(test_label)
    num_zeros_tests = len(test_label) - num_ones_tests 

    mask_train = np.array([True for _ in range(train_edge.shape[0])])


    if num_ones_trains > num_zeros_trains:
        index_ones = np.where(train_label==1)
        index_ones = [t for t in index_ones[0]]
        removed_index_ones = index_ones[num_zeros_trains:]
        mask_train[removed_index_ones] = False 
    elif num_zeros_trains > num_ones_trains:
        index_zeros = np.where(train_label==0)
        index_zeros = [t for t in index_zeros[0]]
        removed_index_zeros = index_zeros[num_ones_trains:]
        mask_train[removed_index_zeros] = False 
    
    train_edge = train_edge[mask_train]
    train_label = train_label[mask_train]

    smallest_of_val_test = min([num_ones_tests, num_zeros_tests, num_ones_vals, num_zeros_vals])

    mask_val_one = np.array([True for _ in range(val_edge.shape[0])])
    mask_val_zero = np.array([True for _ in range(val_edge.shape[0])])
    mask_test_one = np.array([True for _ in range(test_edge.shape[0])])
    mask_test_zero = np.array([True for _ in range(test_edge.shape[0])])
    if num_ones_vals > smallest_of_val_test:
        index_ones = np.where(val_label==1)
        index_ones = [t for t in index_ones[0]]
        removed_index_ones = index_ones[smallest_of_val_test:]
        mask_val_one[removed_index_ones] = False
    if num_zeros_vals > smallest_of_val_test:
        index_zeros = np.where(val_label==0)
        index_zeros = [t for t in index_zeros[0]]
        removed_index_zeros = index_zeros[smallest_of_val_test:]
        mask_val_zero[removed_index_zeros] = False
    if num_ones_tests > smallest_of_val_test:
        index_ones = np.where(test_label==1)
        index_ones = [t for t in index_ones[0]]
        removed_index_ones = index_ones[smallest_of_val_test:]
        mask_test_one[removed_index_ones] = False
    if num_zeros_tests > smallest_of_val_test:
        index_zeros = np.where(test_label==0)
        index_zeros = [t for t in index_zeros[0]]
        removed_index_zeros = index_zeros[smallest_of_val_test:]
        mask_test_zero[removed_index_zeros] = False
    
    mask_val = np.logical_and(mask_val_one, mask_val_zero)
    mask_test = np.logical_and(mask_test_one, mask_test_zero)

    val_edge = val_edge[mask_val]
    val_label = val_label[mask_val]

    test_edge = test_edge[mask_test]
    test_label = test_label[mask_test]

    return train_edge, val_edge, test_edge, train_label, val_label, test_label 
    

def process_data_splits(normalize=True, split_num=""):
    if normalize:
        root = f'./normalized_edge_dataSplits{split_num}'
    else:
        root = './dataSplits'
    make_dirs(root)
    for label_col in meaingfulcol_name_list:
        save_exp = os.path.join(root, label_col)
        make_dirs(save_exp)
        edge, label = load_brain_data(label_col=label_col, normalize=normalize)
        label = np.array(label)
        np.save(os.path.join(save_exp, 'full_edge.npy'), edge)
        np.save(os.path.join(save_exp, 'full_label.npy'), label)

        train_edge, train_label, val_edge, val_label, test_edge, test_label = train_val_test_split(edge, label)
        train_edge, val_edge, test_edge, train_label, val_label, test_label = make_equal_split(train_edge, val_edge, test_edge, train_label, val_label, test_label)

        count_distribution_label(label, attribute='full_label')
        count_distribution_label(train_label, attribute='train_label')
        count_distribution_label(val_label, attribute='val_label')
        count_distribution_label(test_label, attribute='test_label') 

        np.save(os.path.join(save_exp, 'train_edge.npy'), train_edge)
        np.save(os.path.join(save_exp, 'train_label.npy'), train_label)
        np.save(os.path.join(save_exp, 'val_edge.npy'), val_edge)
        np.save(os.path.join(save_exp, 'val_label.npy'), val_label)
        np.save(os.path.join(save_exp, 'test_edge.npy'), test_edge)
        np.save(os.path.join(save_exp, 'test_label.npy'), test_label)


np.random.seed(20003)
process_data_splits(split_num="3")