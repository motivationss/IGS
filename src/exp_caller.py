import time 
import datetime
from train import * 
from train_gradient import *
from train_with_mask import *

def check_args(args):
    assert sum([args.generate_individual_mask_all, args.generate_individual_mask_label_specific]) == 1 
    assert sum([args.sum_unified_mask, args.absolute_sum_unified_mask, args.average_unified_mask, args.max_unified_mask, args.squared_sum_unified_mask]) == 1 
    assert sum([args.metaMask_0, args.metaMask_1, args.metaMask_Avg, args.metaMask_Sum, args.metaMask_AbSum, args.metaMask_Max]) == 1 
    assert sum([args.use_original_edge_mask, args.use_symmetric_edge_mask]) == 1
     
def gnn_during_training(args):

    if args.verbose:
        start_time = time.time()
        
        if args.add_indicator_matrix:
            print("You are using Indicator matrix!")
            
        if args.load_from_Saliency:
            print("You are loading from Saliency!")
        if args.xavier_normal_init:
            print("You are doing xavier_normal_init!")
            
        if args.sigmoid_after_mask:
            print("you are doing sigmoid regualrization on edge mask!")
        elif args.l1_after_mask:
            print("You are doing l1 regularization on edge mask!")
        else:
            print("No Additional Loss Regularization!")
    
    args.save_exp_name = f"pruneThreshold_{args.prune_threshold}_numConvLayers_{args.num_conv_layers}_hiddenChannels_{args.hidden_channels}_lr_{args.lr}_dropout_{args.dropout}_weightDecay_{args.weight_decay}"    
    print("num_pruning_process: ", args.num_pruning_processes)
    print("\n")
    validation_loss = []
    test_accs = []
    for curr_process in range(args.num_pruning_processes):
        print(f"[Current idx {curr_process} Run]")
        gnn_mask_with_training(args) # Edge Mask
        args.curr_idx += 1 
        end_pruning = time.time()
        
        if args.method == 'IGS':
            curr_val_loss, curr_test_acc = dense_training(args, gnn_explain_training=False)
            Gradient_iteration(args)
        
        validation_loss.append(curr_val_loss)
        test_accs.append(curr_test_acc)
        
        if args.verbose:
            end_time = time.time()
            print(f"Current round of pruning takes cumulative of: {str(datetime.timedelta(seconds=(end_pruning - start_time)))}")
            print(f"Current round of pruning plus training takes cumulative of: {str(datetime.timedelta(seconds=(end_time - start_time)))}")
        print("\n")
    
    validation_loss = np.array(validation_loss)
    smallest_index = np.argmin(validation_loss)
    test_acc = test_accs[smallest_index]
    return test_acc 