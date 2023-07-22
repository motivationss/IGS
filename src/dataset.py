from torch_geometric.data import InMemoryDataset
import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


from utils import process_regression_into_classes, process_regression_label_into_binary_heads_tails

meaingfulcol_name_list = ["PicSeq_AgeAdj", "CardSort_AgeAdj", "Flanker_AgeAdj",
"ReadEng_AgeAdj", "PicVocab_AgeAdj", "ListSort_AgeAdj"]

class MyBrainDataset(InMemoryDataset):
    def __init__(self, args, dimension=100, split="train", root= 'data', 
    transform= None, pre_transform= None, pre_filter= None):
        # Label is the ground truth index-column we want to model towards
        label_col = args.label_col
        self.dimension = dimension
        root = os.path.join(root, args.method, f"data_split_{args.dataSplit}",split) 
        
        # # for temp use 
        # if args.method == 'GNNExplainer':
        #     root = os.path.join('data', 'GNNExplainer', split)
        
        root_exp = './data'
        exp_name = os.path.join(root_exp, f'normalized_edge_dataSplits{args.dataSplit}', label_col)
        self.curr_idx = args.curr_idx
        if args.curr_idx > 0:
            exp_save_name = f"pruned_edges{args.curr_idx - 1}.npy"
            print(f"You are using pruned edge {exp_save_name} for {split}!")
            # save_sub_dir = os.path.join("./data/pruned_edges", args.save_exp_name)
            save_sub_dir = os.path.join(args.exp_name, "pruned_edges", args.save_exp_name)
            if not os.path.exists(save_sub_dir):
                os.makedirs(save_sub_dir)
            save_dir = os.path.join(save_sub_dir, f"{split}_{exp_save_name}")
            # with open(save_dir, 'rb') as f:
            edge = np.load(save_dir)
        else:
            print(f"load data from {exp_name}")
            if split == "train":
                edge = np.load(os.path.join(exp_name, 'train_edge.npy')) 
            elif split == "val":
                edge = np.load(os.path.join(exp_name, 'val_edge.npy'))
            elif split == "test":
                edge = np.load(os.path.join(exp_name, 'test_edge.npy'))
            else:
                print("Please enter \{train, val, test\} for split, your prompt was incorrect!")
                raise NotImplementedError 

        self.edge = edge 
        if split == "train":
            self.label = np.load(os.path.join(exp_name, 'train_label.npy')) 
        elif split == "val":
            self.label = np.load(os.path.join(exp_name, 'val_label.npy'))
        elif split == "test":
            self.label = np.load(os.path.join(exp_name, 'test_label.npy'))

        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def __len__(self) -> int:
        return len(self.label)

    @property
    def get_edge(self):
        return self.edge 
    
    @property
    def get_label(self):
        return self.label

    def process(self):
        data_list = []
        
        for idx in range(self.edge.shape[0]):

            curr_edge = self.edge[idx, :, :]
            # if self.curr_idx > 0:
            curr_edge_index, curr_edge_weight = dense_to_sparse(torch.tensor(curr_edge))
            curr_edge_weight = curr_edge_weight.float()
            curr_G = Data(x=torch.eye(self.dimension), edge_index=curr_edge_index, edge_weight=curr_edge_weight,y = torch.LongTensor([int(self.label[idx])]))

            # curr_G = from_networkx(nx.from_numpy_matrix(curr_edge))
            # curr_G.edge_weight = curr_G['weight']
            # curr_G['x'] = torch.eye(self.dimension)
            # curr_y = torch.LongTensor([int(self.label[idx])])  # remove [[self.label[idx]]] so its now one-dim
            # curr_G['y'] = curr_y
            data_list.append(curr_G)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class Subset(torch.utils.data.Dataset):
    """
    Subsets the BrainDataset while preserving the original indexing
    """

    def __init__(self, dataset, indices) -> None:
        super().__init__() 
        self.dataset = dataset 
        self.indices = indices 

        self.edge = self.get_edge()
        self.label = self.get_label()

    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def get_edge(self):
        edge = self.dataset.get_edge[self.indices]
        return edge 
    
    def get_label(self):
        label = self.dataset.get_label[self.indices]
        return label 
