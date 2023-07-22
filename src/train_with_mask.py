import numpy as np
import pandas as pd 
import os 
import torch 

from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
from torch_geometric.utils import from_networkx
from model import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GIN, DenseGCNConv
from utils import *

from dataset import MyBrainDataset, Subset
from torch_geometric.loader import DataLoader
import os

from numpy.linalg import eigh
from utils import *

@torch.no_grad()
def test_dense(model, batch_x, batch_edge, batch_label, criterion, args):
    model.eval()

    out = model.model_forward(batch_x, batch_edge)
    y_pred = out.argmax(dim=-1)

    loss = model.__loss__(out, batch_label)
    # loss = cross_entropy(out, batch_label.view(-1).long())

    return torch.eq(batch_label.view(-1).int(), y_pred).sum().item(), len(y_pred), loss 


def prune_lowest_percent_non_zero_entries(matrix, prune_threshold=0.3):
    assert matrix.ndim == 2  
    assert isinstance(matrix, torch.Tensor)
    matrix_copy = matrix.clone()
    # print( "matrix: ", matrix_copy)
    indices_non_zero = torch.nonzero(matrix_copy)
    nonzero_entries = matrix_copy[indices_non_zero[:,0], indices_non_zero[:,1]]
    _, sort_indices = torch.sort(nonzero_entries)
    lowest_percent_indices =  indices_non_zero[sort_indices[:int(prune_threshold * len(nonzero_entries))]]
    matrix_copy[lowest_percent_indices[:, 0], lowest_percent_indices[:, 1]] = 0 
    return matrix_copy

def gnn_mask_with_training(args):
    
    outer_exp_name = f"./results"
    dataset = args.dataset 
    label = args.label_col
    pruning_mode = True 
    

    model = args.model 
    if args.method == "IGS":
        name1, name2, name3 = transform_names(args)
        args.pruning_mode = f"indi_{name1}_uni_{name2}_meta_{name3}"
    split = f"_split_{args.dataSplit}"
    args.pruning_mode += split

    if pruning_mode:    
        exp_name = os.path.join(outer_exp_name, dataset, model, label, args.method, args.pruning_mode)
    else:
        exp_name = os.path.join(outer_exp_name, dataset, model, label)

    args.exp_name = exp_name 
    

    data_directory = f"./data/normalized_edge_dataSplits{args.dataSplit}/{args.label_col}"
    y_train = np.load(os.path.join(data_directory, "train_label.npy"))
    y_val = np.load(os.path.join(data_directory, "val_label.npy"))
    y_test = np.load(os.path.join(data_directory, "test_label.npy"))

    if args.curr_idx == 0:
        train_adj = np.load(os.path.join(data_directory, "train_edge.npy"))
        val_adj = np.load(os.path.join(data_directory, "val_edge.npy"))
        test_adj = np.load(os.path.join(data_directory, "test_edge.npy"))
    else:
        exp_edge_name = f"pruned_edges{args.curr_idx-1}.npy"
        if args.debug_mode:
            print(f"load edge {exp_edge_name} for train/val/test")
        
        edge_sub_dir = os.path.join(exp_name, "pruned_edges", args.save_exp_name)
        train_edge_dir = os.path.join(edge_sub_dir, f"train_{exp_edge_name}")
        val_edge_dir = os.path.join(edge_sub_dir, f"val_{exp_edge_name}")
        test_edge_dir = os.path.join(edge_sub_dir, f"test_{exp_edge_name}")
        # assert os.path.exists(edge_sub_dir)
        train_adj = np.load(train_edge_dir)
        val_adj = np.load(val_edge_dir)
        test_adj = np.load(test_edge_dir)

    
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    train_adj_tensor = torch.from_numpy(train_adj).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    val_adj_tensor = torch.from_numpy(val_adj).float().to(device)


    y_val_tensor = torch.from_numpy(y_val).to(device)
    test_adj_tensor = torch.from_numpy(test_adj).to(device)
    y_test_tensor = torch.from_numpy(y_test).to(device)

    train_set = TensorDataset(train_adj_tensor, y_train_tensor)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_set = TensorDataset(val_adj_tensor, y_val_tensor)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_set = TensorDataset(test_adj_tensor, y_test_tensor)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)


    typical_data, _ = train_set[0]
    


    criterion = torch.nn.CrossEntropyLoss(reduction="mean")  

    save_metamask_name = f"meta_mask{args.curr_idx}.pt"
    save_metamask_dir = os.path.join(exp_name, 'saved_meta_mask', args.save_exp_name)
    if not os.path.exists(save_metamask_dir):
        os.makedirs(save_metamask_dir)
    if args.load_from_previous:
        previous_metamask_dir = os.path.join(save_metamask_dir, f"meta_mask{args.curr_idx - 1}.pt")
    elif args.load_from_Saliency:
        save_saliency_subexp = os.path.join(exp_name, 'saved_saliency_maps', args.save_exp_name)
        previous_metamask_dir = os.path.join(save_saliency_subexp, f"saliency_map{args.curr_idx}.pth")
    else:
        previous_metamask_dir = ""
    save_metamask_dir = os.path.join(save_metamask_dir, save_metamask_name)
    
    # previous_metamask_dir = os.path.join(save_metamask_dir, f"meta_mask{args.curr_idx - 1}.pt")
    # save_metamask_dir = os.path.join(save_metamask_dir, save_metamask_name)
    
    print("* Current GNN Training with Mask...")
    
    for _ in range(args.num_processes):

        if args.model == "PlainGCN":
            
            model = Dense_mask_training_GCN(typical_data.shape[1], args.hidden_channels,
                        args.num_conv_layers, args.dropout, device=device, args=args, 
                        previous_metamask_dir = previous_metamask_dir, num_classes=2).to(device)
            for m in model.modules():
                if isinstance(m, DenseGCNConv):
                    torch.nn.init.kaiming_normal_(m.lin.weight, nonlinearity='relu')
                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif args.model == "GIN":
            model = Dense_mask_training_GIN(typical_data.shape[1], args.hidden_channels,
                        args.num_conv_layers, args.dropout, device=device, args=args, 
                        previous_metamask_dir = previous_metamask_dir, num_classes=2).to(device)
        elif args.model == "GraphSage":
            model = Dense_mask_training_GraphSage(typical_data.shape[1], args.hidden_channels,
                        args.num_conv_layers, args.dropout, device=device, args=args, 
                        previous_metamask_dir = previous_metamask_dir, num_classes=2).to(device)
        elif args.model == "GraphConv":
            model =  Dense_mask_training_GraphConv(typical_data.shape[1], args.hidden_channels,
                        args.num_conv_layers, args.dropout, device=device, args=args, 
                        previous_metamask_dir = previous_metamask_dir, num_classes=2).to(device)
        else:
            raise NotImplementedError
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        
        
        smallest_val_loss = float('inf')
        smallest_val_loss_corresponding_test_acc = 0

         
        meta_mask = model.get_mask()
        
        
        
        
        for epoch in range(args.num_epochs):
            test_correct, test_total = 0, 0 
            train_correct, train_total = 0, 0
            val_correct, val_total = 0, 0 

            val_loss = 0
            test_loss = 0
    
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                batch_edge, batch_label = batch 
                batch_size = batch_edge.shape[0]
                batch_x = torch.eye(100).repeat(batch_size, 1, 1).to(device)

                optimizer.zero_grad()
                loss = model(batch_x, batch_edge, batch_label)
                loss.backward()
                
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
                optimizer.step()


            for batch in test_loader:
                batch_edge, batch_label = batch 
                batch_edge = batch_edge.float()
                batch_size = batch_edge.shape[0]
                batch_x = torch.eye(100).repeat(batch_size, 1, 1).to(device)

                curr_correct, curr_total, curr_testLoss = test_dense(model=model, batch_x=batch_x,
                batch_edge=batch_edge, criterion=criterion, args=args, batch_label=batch_label)

                test_correct += curr_correct
                test_total += curr_total
                test_loss += curr_testLoss

            for batch in val_loader:
                batch_edge, batch_label = batch 
                batch_size = batch_edge.shape[0]
                batch_x = torch.eye(100).repeat(batch_size, 1, 1).to(device)
                curr_correct, curr_total, curr_valLoss = test_dense(model=model, batch_x=batch_x,
                batch_edge=batch_edge, criterion=criterion, args=args, batch_label=batch_label)
                val_correct += curr_correct
                val_total += curr_total
                val_loss += curr_valLoss
            for batch in train_loader:
                batch_edge, batch_label = batch 
                batch_size = batch_edge.shape[0]
                batch_x = torch.eye(100).repeat(batch_size, 1, 1).to(device)

                curr_correct_train, curr_total_train, _ = test_dense(model=model, batch_x=batch_x,
                batch_edge=batch_edge, criterion=criterion, args=args, batch_label=batch_label)
                train_correct += curr_correct_train
                train_total += curr_total_train
            
            training_acc = train_correct/train_total
            test_acc = test_correct/test_total
            val_acc = val_correct/val_total
            if args.verbose:
                print(f"Epoch {epoch}: train_acc {training_acc:.4g}, val_acc {val_acc:.4g}, test_acc {test_acc:.4g}, val_loss {val_loss:.4g}, train_loss {train_loss:.4g}")
            


                
            if val_loss < smallest_val_loss: 
                smallest_val_loss_corresponding_test_acc = test_acc 
                smallest_val_loss_epoch = epoch 
                smallest_val_loss = val_loss 

                epochs_without_improvement = 0
                meta_mask = model.get_mask().detach()
                if args.use_symmetric_edge_mask:
                    meta_mask = (meta_mask + meta_mask.T) 
                else:
                    meta_mask = meta_mask

                if args.save_model and args.save_exp_name is not None:
                    assert args.method is not None and args.pruning_mode is not None
                    save_sub_dir = os.path.join(exp_name, "model", args.save_exp_name)
                    if not os.path.exists(save_sub_dir):
                        os.makedirs(save_sub_dir)
                    save_name = f"best_val_model{args.curr_idx}.pth"
                    save_dir = os.path.join(save_sub_dir, save_name)
                    torch.save(model, save_dir)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience_epochs:
                break
    
        print(f"Smallest Validation Loss Corresponding Test Accuracy {smallest_val_loss_corresponding_test_acc}, Smallest Validation Loss {smallest_val_loss}")

    if args.debug_mode:
        print(f"metamask saved to {save_metamask_dir}!")
    torch.save(meta_mask, save_metamask_dir)
    
    if args.add_indicator_matrix:
        meta_mask = meta_mask.cpu()
        indicator_exp = os.path.join(exp_name, 'saved_indicator_matrix', args.save_exp_name)
        # print("exp name: ", exp_name)
        if not os.path.exists(indicator_exp):
            os.makedirs(indicator_exp)
        if args.curr_idx == 0:
            # Add Indicator function 
            indicator = torch.ones_like(meta_mask)
        else:
            indicator = torch.load(os.path.join(indicator_exp, f"indicator_{args.curr_idx - 1}.pth"))
        meta_mask = meta_mask * indicator
        print("Number of non-zeros in edge mask before: ", torch.count_nonzero(meta_mask))
        pruned_meta_mask = prune_lowest_percent_non_zero_entries(meta_mask, args.prune_threshold)
        print("Number of non-zeros in edge mask after this round of prunig: ", torch.count_nonzero(pruned_meta_mask))
        pruned_meta_mask[pruned_meta_mask!=0] = 1 
        meta_mask = pruned_meta_mask
        if args.debug_mode:
            print(f"Current Indicator matrix saved to {indicator_exp}")
        torch.save(meta_mask,os.path.join(indicator_exp, f"indicator_{args.curr_idx}.pth"))
        meta_mask = meta_mask.numpy()
    else:
        meta_mask = meta_mask.cpu().numpy()
        threshold_value = np.quantile(meta_mask, args.prune_threshold).item()
        meta_mask = np.where(meta_mask < threshold_value, 0, 1)
    
    pruned_train_adj = np.multiply(train_adj, meta_mask)
    pruned_val_adj = np.multiply(val_adj, meta_mask)
    pruned_test_adj =np.multiply(test_adj, meta_mask)
    
    pruned_train_adj = normalize_edge(pruned_train_adj)
    pruned_val_adj = normalize_edge(pruned_val_adj)
    pruned_test_adj = normalize_edge(pruned_test_adj)
    

    exp_saved_name = f"pruned_edges{args.curr_idx}.npy"
    
    if args.debug_mode:
        print(f"{exp_saved_name} saved for train/val/test!")
    
    saved_sub_dir = os.path.join(exp_name, 'pruned_edges', args.save_exp_name)
    if not os.path.exists(saved_sub_dir):
        os.makedirs(saved_sub_dir)
    train_saved_dir = os.path.join(saved_sub_dir, f"train_{exp_saved_name}")
    test_saved_dir = os.path.join(saved_sub_dir, f"test_{exp_saved_name}")
    val_saved_dir = os.path.join(saved_sub_dir, f"val_{exp_saved_name}")

    np.save(train_saved_dir, pruned_train_adj)
    np.save(test_saved_dir, pruned_test_adj)
    np.save(val_saved_dir, pruned_val_adj)
    if args.debug_mode:
        print("save train dir: ", train_saved_dir)
        print("save test dir: ", test_saved_dir)
        print("\n")