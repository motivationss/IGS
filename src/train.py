import os 
import numpy as np 
import torch 
from torch.utils.data import DataLoader, TensorDataset
from model import * 
from utils import * 

@torch.no_grad()
def test_dense(model, batch_x, batch_edge, batch_label, criterion, args, ):
    model.eval()

    out = model(batch_x, batch_edge)
    y_pred = out.argmax(dim=-1)
    loss = criterion(out, batch_label.view(-1).long())

    return torch.eq(batch_label.view(-1).int(), y_pred).sum().item(), len(y_pred), loss 


def dense_training(args, gnn_explain_training=False):
    outer_exp_name = f"./results"
    dataset = args.dataset 
    label = args.label_col

    split = f"_split_{args.dataSplit}"

    if args.save_model and args.save_exp_name is not None:
        # assert args.pruning_method is not None and args.pruning_mode is not None
        if args.method == "IGS":
            name1, name2, name3 = transform_names(args)
            args.pruning_mode = f"indi_{name1}_uni_{name2}_meta_{name3}"
        pruning_mode = True 
    else:
        pruning_mode = False 
    if args.pruning_mode is not None:
        args.pruning_mode += split 
    model = args.model 
    
    if pruning_mode:
        exp_name = os.path.join(outer_exp_name, dataset, model, label, args.method, args.pruning_mode)
        
    else:
        exp_name = os.path.join(outer_exp_name, dataset, model, label)

    args.exp_name = exp_name 

    data_directory = f"./data/normalized_edge_dataSplits{args.dataSplit}/{args.label_col}"
    
    if args.curr_idx == 0 and args.debug_mode:
        print(f"DenseTraining Load data from {data_directory}!")
    y_train = np.load(os.path.join(data_directory, "train_label.npy"))
    y_val = np.load(os.path.join(data_directory, "val_label.npy"))
    y_test = np.load(os.path.join(data_directory, "test_label.npy"))

    if args.curr_idx == 0:
        train_adj = np.load(os.path.join(data_directory, "train_edge.npy"))
        val_adj = np.load(os.path.join(data_directory, "val_edge.npy"))
        test_adj = np.load(os.path.join(data_directory, "test_edge.npy"))
    else:
        exp_edge_name = f"pruned_edges{args.curr_idx-1}.npy"
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
    
    print("* Currently Normal Training...")

    if args.model == "PlainGCN":
        model = Dense_GCN(typical_data.shape[1], args.hidden_channels,
                    args.num_conv_layers, args.dropout, 2).to(device)
        for m in model.modules():
            if isinstance(m, DenseGCNConv):
                torch.nn.init.kaiming_normal_(m.lin.weight, nonlinearity='relu')
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif args.model == "GIN":
        model = Dense_GIN(typical_data.shape[1], args.hidden_channels,
                    args.num_conv_layers, args.dropout, 2).to(device)
    elif args.model == "GraphSage":
        model = Dense_GraphSage(typical_data.shape[1], args.hidden_channels,
                    args.num_conv_layers, args.dropout, 2).to(device)
    elif args.model == "GraphConv":
        model = Dense_GraphConv(typical_data.shape[1], args.hidden_channels,
                    args.num_conv_layers, args.dropout, 2).to(device)
    else:
        raise NotImplementedError("Check your model argumets")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    
    smallest_val_loss = float('inf')
    smallest_val_loss_corresponding_test_acc = 0

    for epoch in range(args.num_epochs):
        test_correct, test_total = 0, 0 
        train_correct, train_total = 0, 0
        val_correct, val_total = 0, 0 

        val_loss = 0
        train_loss = 0 
        
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch_edge, batch_label = batch 
            batch_size = batch_edge.shape[0]
            batch_x = torch.eye(100).repeat(batch_size, 1, 1).to(device)

            optimizer.zero_grad()
            out = model(batch_x, batch_edge)
            loss = criterion(out, batch_label.view(-1).long())
            loss.backward()
            optimizer.step()
            train_loss += loss 
            
            # if args.verbose:
            #     print(f"Epoch {epoch} Batch {batch_idx}: training loss {loss}")

        for batch in test_loader:
            batch_edge, batch_label = batch 
            batch_edge = batch_edge.float()
            batch_size = batch_edge.shape[0]
            batch_x = torch.eye(100).repeat(batch_size, 1, 1).to(device)

            curr_correct, curr_total, curr_testLoss = test_dense(model=model, batch_x=batch_x,
            batch_edge=batch_edge, criterion=criterion, args=args, batch_label=batch_label)

            test_correct += curr_correct
            test_total += curr_total

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
            smallest_val_loss = val_loss 

            epochs_without_improvement = 0

            if args.save_model and args.save_exp_name is not None and not gnn_explain_training:
                assert args.method is not None and args.pruning_mode is not None
                save_sub_dir = os.path.join(exp_name, "model", args.save_exp_name)
                # save_sub_dir = f"./model/{args.save_exp_name}"
                if not os.path.exists(save_sub_dir):
                    os.makedirs(save_sub_dir)
                save_name = f"best_val_model{args.curr_idx}.pth"
                save_dir = os.path.join(save_sub_dir, save_name)
                # print("Dense Model saved to ", save_dir)
                torch.save(model, save_dir)
    
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience_epochs:
            break
    
    if args.curr_idx != 0:
        smallest_val_loss = smallest_val_loss.item()
    print(f"Smallest Validation Loss Corresponding Test Accuracy {smallest_val_loss_corresponding_test_acc}, Smallest Validation Loss {smallest_val_loss}")
    
    return smallest_val_loss, smallest_val_loss_corresponding_test_acc