from dataclasses import dataclass, field
from typing import Optional

@dataclass 
class IGSConfig:
    """
    Arguments that IGS specifically have for training 
    """
    
    mask_function: Optional[str] = field(
        default="Sigmoid",
        metadata={"help": "Probability function acting on the Mask"},
    )
    
    xavier_unif_init: Optional[bool] = field(
        default=False,
        metadata={"help": "using xavier uniform to init mask"},
    )
    
    load_from_previous: Optional[bool] = field(
        default=False, 
        metadata={"help": "load from previous iteration to init mask"}
    )
    
    load_from_Saliency: Optional[bool] = field(
        default=True,
        metadata={"help": "load with gradients"}
    )
    
    generate_individual_mask_all: Optional[bool] = field(
        default=True, 
        metadata={"help": "individual gradient mask for both class"} 
    )
    
    metaMask_Sum: Optional[bool] = field(
        default=True,
        metadata={"help": "take sum of of individual mask"} 
    )
    
    absolute_sum_unified_mask: Optional[bool] = field(
        default=True,
        metadata={"help": "take absolute sum of unified mask"}
    )
    
    add_indicator_matrix: Optional[bool] = field(
        default=True,
        metadata={"help": "avoid pruning same entires over iterations"}
    )
    
    sigmoid_after_mask: Optional[bool] = field(
        default=True,
        metadata={"help": "loss regularization using sigmoid"}
    )
    
    l1_after_mask: Optional[bool] = field(
        default=False,
        metadata={"help": "loss regularization using l1"}
    )
    
    use_original_edge_mask: Optional[bool] = field(
        default=False,
        metadata={"help": "original edge mask"}
    )
    
    use_symmetric_edge_mask: Optional[bool] = field(
        default=True,
        metadata={"help": "use symmetrized edge mask"}
    )
    
    save_model: Optional[bool] = field(
        default=True,
        metadata={"help": "whether saving model"}
    )