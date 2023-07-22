import argparse
from utils import *
from train import * 
import numpy as np
from exp_caller import *
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.configurations import * 

def run_exp(args):
    print(f"You are using {args.model} as the backbone!")
    if args.method == 'Normal':
       _, test_acc = dense_training(args)
    elif args.method == "IGS":
       test_acc = gnn_during_training(args)
    else:
        raise NotImplementedError

    return test_acc

def meta_exp(args):
    seed_everything(args.seed)
    
    # Configurations 
    if args.method == "IGS":
        config = IGSConfig()
        config_dict = vars(config)
                
        for arg_name, arg_value in config_dict.items():
            setattr(args, arg_name, arg_value)
    print("args: ", args)
    
    if args.dataSplit == "general":
        test_acc = []
        for args.dataSplit in ["1", "2", "3", "4"]:
            print(f"##################################### current split: {args.dataSplit} #####################################")
            curr_test_acc = run_exp(args)
            args.curr_idx = 0
            test_acc.append(curr_test_acc)
            print("\n")
        print("Training Accuracy: ", test_acc)
        print(f"Average of Test Accuracy {np.average(test_acc) * 100 :.4f}, Std of Test Accuracy {np.std(test_acc)*100:4f}")
    else:
        assert args.dataSplit in ["1", "2", "3", "4"], "Please Double Check your dataSplit input"
        test_acc = run_exp(args)
        print("Test Accuracy: ", test_acc)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCN baseline')

    # Specify method, dataset, and target_task at first 
    parser.add_argument('--method', type=str, default='Normal', 
    choices=["Normal", "IGS"])
    parser.add_argument("--dataset", type=str, default="BrainDataset")
    parser.add_argument("--label_col", type=str, default="PicVocab_AgeAdj",
    choices=["PicVocab_AgeAdj", "ReadEng_AgeAdj", "PicSeq_AgeAdj", 
            "ListSort_AgeAdj", "CardSort_AgeAdj", "Flanker_AgeAdj"],
    help="task of interest")
    parser.add_argument("--dataSplit", type=str, default="general")
    
    # General Args for experimentation 
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_conv_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience_epochs', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='./log_files')
    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--model", type=str, default="PlainGCN", 
                        choices=["PlainGCN", "GIN", "GraphSage", "GraphConv"],
                        help="model backbone")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_name", type=str, default="model.pth")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--curr_idx", type=int, default=0)
    parser.add_argument("--save_exp_name", type=str, default=None)
    parser.add_argument("--pruning_method", type=str, default=None)
    parser.add_argument("--pruning_mode", type=str, default=None)
   
    # Gradient Pruning Args
    parser.add_argument("--prune_threshold", type=float, default=0.05)

    # generating individual
    parser.add_argument("--generate_individual_mask_all", action="store_true", default=False)
    parser.add_argument("--generate_individual_mask_label_specific", action="store_true", default=False)

    # generate Unified-Mask (0 and 1)
    parser.add_argument("--sum_unified_mask", action="store_true", default=False)
    parser.add_argument("--absolute_sum_unified_mask", action="store_true", default=False)
    parser.add_argument("--average_unified_mask", action="store_true", default=False) 
    parser.add_argument("--max_unified_mask", action="store_true", default=False) 
    parser.add_argument("--squared_sum_unified_mask", action="store_true", default=False)     

    # Combine Meta-Mask:
    parser.add_argument("--metaMask_0", action="store_true", default=False)
    parser.add_argument("--metaMask_1", action="store_true", default=False)
    parser.add_argument("--metaMask_Avg", action="store_true", default=False)
    parser.add_argument("--metaMask_Sum", action="store_true", default=False)
    parser.add_argument("--metaMask_AbSum", action="store_true", default=False)
    parser.add_argument("--metaMask_Max", action="store_true", default=False)

    # Generating Individual: (1) All (2) Label-Specific  [Gradient 0; Gradient 1]
    # Unified-Mask : (1) Average (2) Sum (3) Absolute Sum (4) Max 
    # Combine Meta-Mask : (1) Use only One MetaMask (2) Average (3) Sum (4) Absolute Sum (5) Max

    # Trained Edge mask 
    parser.add_argument("--l1_mask_training", action="store_true", default=False)
    parser.add_argument("--l2_mask_training", action="store_true", default=False)
    parser.add_argument("--sigmoid_mask_training", action="store_true", default=False)
    parser.add_argument("--mask_reLU", action="store_true", default=False)
    parser.add_argument("--regularizor_mask_training", type=float, default=1e-4)

    parser.add_argument("--num_pruning_processes", type=int, default=55)

    parser.add_argument("--mask_function", type=str, default="Sigmoid")
    parser.add_argument("--l1_after_mask", action="store_true", default=False)
    parser.add_argument("--use_original_edge_mask", action="store_true", default=False)
    parser.add_argument("--use_symmetric_edge_mask", action="store_true", default=False)

    parser.add_argument("--xavier_unif_init", action="store_true", default=False)
    parser.add_argument("--xavier_normal_init", action="store_true", default=False)
    parser.add_argument("--penalize_original_mask", action="store_true", default=False)
    parser.add_argument("--add_indicator_matrix", action="store_true", default=False)
    parser.add_argument("--load_from_previous", action="store_true", default=False)
    parser.add_argument("--load_from_Saliency", action="store_true", default=False)
    parser.add_argument("--sigmoid_after_mask", action="store_true", default=False)
    args = parser.parse_args()
    # print(args)
    meta_exp(args)