import torch 
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GIN
from torch_geometric.nn import DenseGINConv, DenseSAGEConv, DenseGraphConv, DenseGCNConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
from torch.nn import LayerNorm
import numpy as np 



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_conv_layers,
                 dropout, num_classes=2):
        super(GCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_conv_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))

        self.conv_layer_norms = torch.nn.ModuleList()
        self.conv_layer_norms.append(torch.nn.LayerNorm(100))
        for _ in range(num_conv_layers - 1):
            self.conv_layer_norms.append(LayerNorm(hidden_channels))        

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(hidden_channels, num_classes))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t=None, batch=None, edge_index=None, edge_weight=None):
        
        for (conv, layernorm) in zip(self.convs, self.conv_layer_norms):

                x = layernorm(x) 
                x = conv(x, edge_index=edge_index, edge_weight=edge_weight)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lins[0](x)
        return x

class Dense_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_conv_layers,
                 dropout, num_classes=2):
        super(Dense_GCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            DenseGCNConv(in_channels, hidden_channels))
        for _ in range(num_conv_layers - 1):
            self.convs.append(
                DenseGCNConv(hidden_channels, hidden_channels))

        self.conv_layer_norms = torch.nn.ModuleList()
        self.conv_layer_norms.append(torch.nn.LayerNorm(100))
        for _ in range(num_conv_layers - 1):
            self.conv_layer_norms.append(LayerNorm(hidden_channels))        

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(hidden_channels, num_classes))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        
        for (conv, layernorm) in zip(self.convs, self.conv_layer_norms):
                x = layernorm(x) 
                x = conv(x, adj)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
        
        x = torch.mean(x, dim=1)
        x = self.lins[0](x)
        return x

class Dense_GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_conv_layers,
                 dropout, num_classes=2):
        super(Dense_GIN, self).__init__()

        nn_start = Sequential(Linear(in_channels, hidden_channels))
        r_start = np.random.uniform()
        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append( 
            DenseGINConv(nn=nn_start, eps=r_start, train_eps=True))
        
        
        for _ in range(num_conv_layers - 1):
            nn = Sequential(Linear(hidden_channels, hidden_channels))
            rs = np.random.uniform()
            self.convs.append(
                DenseGINConv(nn, rs, train_eps=True))

        self.conv_layer_norms = torch.nn.ModuleList()
        self.conv_layer_norms.append(torch.nn.LayerNorm(100))
        for _ in range(num_conv_layers - 1):
            self.conv_layer_norms.append(LayerNorm(hidden_channels))        

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(hidden_channels, num_classes))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        
        for (conv, layernorm) in zip(self.convs, self.conv_layer_norms):
                x = layernorm(x) 
                x = conv(x, adj)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
        
        x = torch.mean(x, dim=1)
        x = self.lins[0](x)
        return x

class Dense_GraphSage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_conv_layers,
                 dropout, num_classes=2):
        super(Dense_GraphSage, self).__init__()

        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append( 
            DenseSAGEConv(in_channels, hidden_channels))
        
        
        for _ in range(num_conv_layers - 1):
            self.convs.append(
                DenseSAGEConv(hidden_channels, hidden_channels))

        self.conv_layer_norms = torch.nn.ModuleList()
        self.conv_layer_norms.append(torch.nn.LayerNorm(100))
        for _ in range(num_conv_layers - 1):
            self.conv_layer_norms.append(LayerNorm(hidden_channels))        

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(hidden_channels, num_classes))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        
        for (conv, layernorm) in zip(self.convs, self.conv_layer_norms):
                x = layernorm(x) 
                x = conv(x, adj)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
        
        x = torch.mean(x, dim=1)
        x = self.lins[0](x)
        return x

class Dense_GraphConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_conv_layers,
                 dropout, num_classes=2):
        super(Dense_GraphConv, self).__init__()
        # If we have other feature matrix, then we probably need linear layer here

        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append( 
            DenseGraphConv(in_channels, hidden_channels))
        
        
        for _ in range(num_conv_layers - 1):
            self.convs.append(
                DenseGraphConv(hidden_channels, hidden_channels))

        self.conv_layer_norms = torch.nn.ModuleList()
        self.conv_layer_norms.append(torch.nn.LayerNorm(100))
        for _ in range(num_conv_layers - 1):
            self.conv_layer_norms.append(LayerNorm(hidden_channels))        

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(hidden_channels, num_classes))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        
        for (conv, layernorm) in zip(self.convs, self.conv_layer_norms):
                x = layernorm(x) 
                x = conv(x, adj)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
        
        x = torch.mean(x, dim=1)
        x = self.lins[0](x)
        return x


import os 
class Dense_mask_training_GCN(Dense_GCN):


    def __init__(self, in_channels, hidden_channels, num_conv_layers, dropout, 
    device, args, previous_metamask_dir, num_classes=2, prune_training_mask=0.05):
        super().__init__(in_channels, hidden_channels, num_conv_layers, dropout, num_classes)
    
        self.args = args 
        self.edge_mask = None
        self.hard_edge_mask = None
        self.previous_dir_meta_mask = previous_metamask_dir
        self.device = device 
        self.__set_masks__()

        
        if self.args.add_indicator_matrix:
            if self.args.curr_idx == 0:
                self.indicator_matrix = torch.ones_like(self.edge_mask).to(self.device)
            else:
                outer_exp_name = f"./results"
                exp_name = os.path.join(outer_exp_name, self.args.dataset, self.args.model, self.args.label_col, self.args.method, self.args.pruning_mode)
                indicator_exp = os.path.join(exp_name, 'saved_indicator_matrix', self.args.save_exp_name)
                indicator = torch.load(os.path.join(indicator_exp, f"indicator_{self.args.curr_idx - 1}.pth"))
                self.indicator_matrix = indicator.to(self.device)
    


    def __set_masks__(self, num_nodes=100, x_feat_dim=100, init="normal"):
        N = num_nodes
        F = x_feat_dim

        
        if self.args.xavier_unif_init:
            self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
            torch.nn.init.xavier_uniform_(self.edge_mask)
        elif self.args.xavier_normal_init:
            self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
            torch.nn.init.xavier_normal_(self.edge_mask)
        elif self.args.load_from_previous:
            if self.args.curr_idx != 0:
                self.edge_mask = torch.load(self.previous_dir_meta_mask) 
                    
            else:
                self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
                torch.nn.init.xavier_normal_(self.edge_mask)
        elif self.args.load_from_Saliency:
            if self.args.curr_idx != 0:
                self.edge_mask = torch.load(self.previous_dir_meta_mask)
                self.edge_mask = 0.4 * self.edge_mask / self.edge_mask.max().item() 
                self.edge_mask = torch.nn.Parameter(self.edge_mask, requires_grad=True)
                
            else:
                self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
                torch.nn.init.xavier_normal_(self.edge_mask)

        else:
            self.edge_mask = torch.nn.Parameter(torch.randn(N, N, requires_grad=True))

    def get_mask(self):
        return self.edge_mask 
    

    def __loss__(self, raw_preds, label):
        loss = cross_entropy(raw_preds, label.view(-1).long())

        if self.args.use_original_edge_mask:
            edge_mask = self.edge_mask 
        elif self.args.use_symmetric_edge_mask:
            edge_mask = self.edge_mask + self.edge_mask.T 
        else:
            raise NotImplementedError("Check your args")
        

        if self.args.add_indicator_matrix:
            edge_mask = edge_mask * self.indicator_matrix 
        
        if self.args.l1_after_mask:
            loss = loss + self.args.regularizor_mask_training * torch.norm(edge_mask, p=1)
        elif self.args.sigmoid_after_mask:
            loss = loss + self.args.regularizor_mask_training * edge_mask.sigmoid().sum()
        else:
            loss = loss 
        return loss 

    def forward(self, x, adj, batch_label):
        raw_preds = self.model_forward(x, adj)
        return self.__loss__(raw_preds, batch_label)

    def model_forward(self, x, adj):
        
        if self.args.use_original_edge_mask:
            edge_mask = self.edge_mask 
        elif self.args.use_symmetric_edge_mask:
            edge_mask = self.edge_mask + self.edge_mask.T 
        else:
            raise NotImplementedError("Check your args")

        if self.args.add_indicator_matrix:
            edge_mask = edge_mask * self.indicator_matrix 

        if self.args.mask_function == "Sigmoid":
            edge_mask = edge_mask.sigmoid()
        elif self.args.mask_function == "ReLU":
            edge_mask = F.relu(edge_mask)
        elif self.args.mask_function == "LeakyRelu":
            edge_mask = F.leaky_relu(edge_mask, negative_slope=0.1)
        else:
            raise NotImplementedError
        
        pruned_adj = torch.multiply(adj, edge_mask)        

        for (conv, layernorm) in zip(self.convs, self.conv_layer_norms):
                x = layernorm(x) 
                x = conv(x, pruned_adj)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
        
        x = torch.mean(x, dim=1)
        x = self.lins[0](x)
        return x

class Dense_mask_training_GIN(Dense_GIN):

    def __init__(self, in_channels, hidden_channels, num_conv_layers, dropout, 
    device, args, previous_metamask_dir, num_classes=2, prune_training_mask=0.05):
        super().__init__(in_channels, hidden_channels, num_conv_layers, dropout, num_classes)
    
        self.args = args 
        self.edge_mask = None
        self.hard_edge_mask = None
        self.previous_dir_meta_mask = previous_metamask_dir
        self.device = device 
        self.__set_masks__()

        
        if self.args.add_indicator_matrix:
            if self.args.curr_idx == 0:
                self.indicator_matrix = torch.ones_like(self.edge_mask).to(self.device)
            else:
                outer_exp_name = f"./results"
                exp_name = os.path.join(outer_exp_name, self.args.dataset, self.args.model, self.args.label_col, self.args.method, self.args.pruning_mode)
                indicator_exp = os.path.join(exp_name, 'saved_indicator_matrix', self.args.save_exp_name)
                indicator = torch.load(os.path.join(indicator_exp, f"indicator_{self.args.curr_idx - 1}.pth"))
                self.indicator_matrix = indicator.to(self.device)
    


    def __set_masks__(self, num_nodes=100, x_feat_dim=100, init="normal"):
        N = num_nodes
        F = x_feat_dim

        if self.args.xavier_unif_init:
            self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
            torch.nn.init.xavier_uniform_(self.edge_mask)
        elif self.args.xavier_normal_init:
            self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
            torch.nn.init.xavier_normal_(self.edge_mask)
        elif self.args.load_from_previous:
            if self.args.curr_idx != 0:
                self.edge_mask = torch.load(self.previous_dir_meta_mask) 
            else:
                self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
                torch.nn.init.xavier_normal_(self.edge_mask)
        elif self.args.load_from_Saliency:
            if self.args.curr_idx != 0:
                self.edge_mask = torch.load(self.previous_dir_meta_mask)
                self.edge_mask = 0.4 * self.edge_mask / self.edge_mask.max().item() 
                self.edge_mask = torch.nn.Parameter(self.edge_mask, requires_grad=True)
            else:
                self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
                torch.nn.init.xavier_normal_(self.edge_mask)
        else:
            self.edge_mask = torch.nn.Parameter(torch.randn(N, N, requires_grad=True))

    def get_mask(self):
        return self.edge_mask 
    

    def __loss__(self, raw_preds, label):
        loss = cross_entropy(raw_preds, label.view(-1).long())

        if self.args.use_original_edge_mask:
            edge_mask = self.edge_mask 
        elif self.args.use_symmetric_edge_mask:
            edge_mask = self.edge_mask + self.edge_mask.T 
        else:
            raise NotImplementedError("Check your args")
        

        if self.args.add_indicator_matrix:
            edge_mask = edge_mask * self.indicator_matrix 
        
        if self.args.l1_after_mask:
            loss = loss + self.args.regularizor_mask_training * torch.norm(edge_mask, p=1)
        elif self.args.sigmoid_after_mask:
            loss = loss + self.args.regularizor_mask_training * edge_mask.sigmoid().sum()
        else:
            loss = loss 
        return loss 

    def forward(self, x, adj, batch_label):
        raw_preds = self.model_forward(x, adj)
        return self.__loss__(raw_preds, batch_label)

    def model_forward(self, x, adj):
        if self.args.use_original_edge_mask:
            edge_mask = self.edge_mask 
        elif self.args.use_symmetric_edge_mask:
            edge_mask = self.edge_mask + self.edge_mask.T 
        else:
            raise NotImplementedError("Check your args")
        

        if self.args.add_indicator_matrix:
            edge_mask = edge_mask * self.indicator_matrix  

        if self.args.mask_function == "Sigmoid":
            edge_mask = edge_mask.sigmoid()
        elif self.args.mask_function == "ReLU":
            edge_mask = F.relu(edge_mask)
        elif self.args.mask_function == "LeakyRelu":
            edge_mask = F.leaky_relu(edge_mask, negative_slope=0.1)
        else:
            raise NotImplementedError
       
        pruned_adj = torch.multiply(adj, edge_mask)      

        for (conv, layernorm) in zip(self.convs, self.conv_layer_norms):
                x = layernorm(x) 
                x = conv(x, pruned_adj)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
        
        x = torch.mean(x, dim=1)
        x = self.lins[0](x)
        return x

class Dense_mask_training_GraphSage(Dense_GraphSage):


    def __init__(self, in_channels, hidden_channels, num_conv_layers, dropout, 
    device, args, previous_metamask_dir, num_classes=2, prune_training_mask=0.05):
        super().__init__(in_channels, hidden_channels, num_conv_layers, dropout, num_classes)
    
        self.args = args 
        self.edge_mask = None
        self.hard_edge_mask = None
        self.previous_dir_meta_mask = previous_metamask_dir
        self.device = device 
        self.__set_masks__()
        
        if self.args.add_indicator_matrix:
            if self.args.curr_idx == 0:
                self.indicator_matrix = torch.ones_like(self.edge_mask).to(self.device)
            else:
                outer_exp_name = f"./results"
                exp_name = os.path.join(outer_exp_name, self.args.dataset, self.args.model, self.args.label_col, self.args.method, self.args.pruning_mode)
                indicator_exp = os.path.join(exp_name, 'saved_indicator_matrix', self.args.save_exp_name)
                indicator = torch.load(os.path.join(indicator_exp, f"indicator_{self.args.curr_idx - 1}.pth"))
                self.indicator_matrix = indicator.to(self.device)
    


    def __set_masks__(self, num_nodes=100, x_feat_dim=100, init="normal"):
        N = num_nodes
        F = x_feat_dim

        if self.args.xavier_unif_init:
            self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
            torch.nn.init.xavier_uniform_(self.edge_mask)
        elif self.args.xavier_normal_init:
            self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
            torch.nn.init.xavier_normal_(self.edge_mask)
        elif self.args.load_from_previous:
            if self.args.curr_idx != 0:
                self.edge_mask = torch.load(self.previous_dir_meta_mask) 
                    
            else:
                self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
                torch.nn.init.xavier_normal_(self.edge_mask)
        elif self.args.load_from_Saliency:
            if self.args.curr_idx != 0:
                self.edge_mask = torch.load(self.previous_dir_meta_mask)
                self.edge_mask = 0.4 * self.edge_mask / self.edge_mask.max().item() 
                self.edge_mask = torch.nn.Parameter(self.edge_mask, requires_grad=True)
            else:
                self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
                torch.nn.init.xavier_normal_(self.edge_mask)

        else:
            self.edge_mask = torch.nn.Parameter(torch.randn(N, N, requires_grad=True))

    def get_mask(self):
        return self.edge_mask 
    

    def __loss__(self, raw_preds, label):
        loss = cross_entropy(raw_preds, label.view(-1).long())

        if self.args.use_original_edge_mask:
            edge_mask = self.edge_mask 
        elif self.args.use_symmetric_edge_mask:
            edge_mask = self.edge_mask + self.edge_mask.T 
        else:
            raise NotImplementedError("Check your args")
        

        if self.args.add_indicator_matrix:
            edge_mask = edge_mask * self.indicator_matrix 
        
        if self.args.l1_after_mask:
            loss = loss + self.args.regularizor_mask_training * torch.norm(edge_mask, p=1)
        elif self.args.sigmoid_after_mask:
            loss = loss + self.args.regularizor_mask_training * edge_mask.sigmoid().sum()
        else:
            loss = loss 
        return loss 

    def forward(self, x, adj, batch_label):
        raw_preds = self.model_forward(x, adj)
        return self.__loss__(raw_preds, batch_label)

    def model_forward(self, x, adj):
        
        if self.args.use_original_edge_mask:
            edge_mask = self.edge_mask 
        elif self.args.use_symmetric_edge_mask:
            edge_mask = self.edge_mask + self.edge_mask.T 
        else:
            raise NotImplementedError("Check your args")

        if self.args.add_indicator_matrix:
            edge_mask = edge_mask * self.indicator_matrix

        if self.args.mask_function == "Sigmoid":
            edge_mask = edge_mask.sigmoid()
        elif self.args.mask_function == "ReLU":
            edge_mask = F.relu(edge_mask)
        elif self.args.mask_function == "LeakyRelu":
            edge_mask = F.leaky_relu(edge_mask, negative_slope=0.1)
        else:
            raise NotImplementedError
       
        pruned_adj = torch.multiply(adj, edge_mask) 
         

        for (conv, layernorm) in zip(self.convs, self.conv_layer_norms):
                x = layernorm(x) 
                x = conv(x, pruned_adj)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
        
        x = torch.mean(x, dim=1)
        x = self.lins[0](x)
        return x


class Dense_mask_training_GraphConv(Dense_GraphConv):

    def __init__(self, in_channels, hidden_channels, num_conv_layers, dropout, 
    device, args, previous_metamask_dir, num_classes=2, prune_training_mask=0.05):
        super().__init__(in_channels, hidden_channels, num_conv_layers, dropout, num_classes)
    
        self.args = args 
        self.edge_mask = None
        self.hard_edge_mask = None
        self.previous_dir_meta_mask = previous_metamask_dir
        self.device = device 
        self.__set_masks__()
        
        if self.args.add_indicator_matrix:
            if self.args.curr_idx == 0:
                self.indicator_matrix = torch.ones_like(self.edge_mask).to(self.device)
            else:
                outer_exp_name = f"./results"
                exp_name = os.path.join(outer_exp_name, self.args.dataset, self.args.model, self.args.label_col, self.args.method, self.args.pruning_mode)
                indicator_exp = os.path.join(exp_name, 'saved_indicator_matrix', self.args.save_exp_name)
                indicator = torch.load(os.path.join(indicator_exp, f"indicator_{self.args.curr_idx - 1}.pth"))
                self.indicator_matrix = indicator.to(self.device)
    


    def __set_masks__(self, num_nodes=100, x_feat_dim=100, init="normal"):
        N = num_nodes
        F = x_feat_dim
        
        if self.args.xavier_unif_init:
            self.edge_mask = torch.nn.Parameter(torch.empty(N,N, requires_grad=True))
            torch.nn.init.xavier_uniform_(self.edge_mask)
        elif self.args.xavier_normal_init:
            self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
            torch.nn.init.xavier_normal_(self.edge_mask)
        elif self.args.load_from_previous:
            if self.args.curr_idx != 0:
                self.edge_mask = torch.load(self.previous_dir_meta_mask) 
                    
            else:
                self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
                torch.nn.init.xavier_normal_(self.edge_mask)
        elif self.args.load_from_Saliency:
            if self.args.curr_idx != 0:
                self.edge_mask = torch.load(self.previous_dir_meta_mask)
                self.edge_mask = 0.4 * self.edge_mask / self.edge_mask.max().item() 
                self.edge_mask = torch.nn.Parameter(self.edge_mask, requires_grad=True)
            else:
                self.edge_mask = torch.nn.Parameter(torch.empty(N, N, requires_grad=True))
                torch.nn.init.xavier_normal_(self.edge_mask)

        else:
            self.edge_mask = torch.nn.Parameter(torch.randn(N, N, requires_grad=True))

    def get_mask(self):
        return self.edge_mask 
    

    def __loss__(self, raw_preds, label):
        loss = cross_entropy(raw_preds, label.view(-1).long())

        if self.args.use_original_edge_mask:
            edge_mask = self.edge_mask 
        elif self.args.use_symmetric_edge_mask:
            edge_mask = self.edge_mask + self.edge_mask.T 
        else:
            raise NotImplementedError("Check your args")
        

        if self.args.add_indicator_matrix:
            edge_mask = edge_mask * self.indicator_matrix 
        
        if self.args.l1_after_mask:
            loss = loss + self.args.regularizor_mask_training * torch.norm(edge_mask, p=1)
        elif self.args.sigmoid_after_mask:
            loss = loss + self.args.regularizor_mask_training * edge_mask.sigmoid().sum()
        else:
            loss = loss 
        return loss 

    def forward(self, x, adj, batch_label):
        raw_preds = self.model_forward(x, adj)
        return self.__loss__(raw_preds, batch_label)

    def model_forward(self, x, adj):
        if self.args.use_original_edge_mask:
            edge_mask = self.edge_mask 
        elif self.args.use_symmetric_edge_mask:
            edge_mask = self.edge_mask + self.edge_mask.T 
        else:
            raise NotImplementedError("Check your args")


        if self.args.add_indicator_matrix:
            edge_mask = edge_mask * self.indicator_matrix  

        if self.args.mask_function == "Sigmoid":
            edge_mask = edge_mask.sigmoid()
        elif self.args.mask_function == "ReLU":
            edge_mask = F.relu(edge_mask)
        elif self.args.mask_function == "LeakyRelu":
            edge_mask = F.leaky_relu(edge_mask, negative_slope=0.1)
        else:
            raise NotImplementedError
        
        pruned_adj = torch.multiply(adj, edge_mask)        
        
        for (conv, layernorm) in zip(self.convs, self.conv_layer_norms):
                x = layernorm(x) 
                x = conv(x, pruned_adj)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
        
        x = torch.mean(x, dim=1)
        x = self.lins[0](x)
        return x