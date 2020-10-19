from typing import Tuple

from torch import Tensor
from torch.nn import Module
from torch_geometric.utils import dropout_adj


class PropagationDropout(Module):
    def __init__(self,
                 p: float,
                 ):
        super().__init__()
        self.p = p

    def forward(self,
                edge_index: Tensor,
                edge_weight: Tensor,
                ) -> Tuple[Tensor, Tensor]:
        edge_index, edge_weight = dropout_adj(edge_index=edge_index,
                                              edge_attr=edge_weight,
                                              p=self.p,
                                              training=self.training,
                                              )
        return edge_index, edge_weight
