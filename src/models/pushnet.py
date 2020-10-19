import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Linear, Sequential, ReLU, Dropout
from torch.nn.functional import log_softmax
from torch.nn.init import _calculate_fan_in_and_fan_out, kaiming_uniform_, uniform_
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from models.utils import PropagationDropout


class PushNetConv(MessagePassing):
    def __init__(self,
                 cached: bool,
                 batch_size_messages: Optional[int] = None,
                 ):
        super().__init__(aggr='add',
                         flow='target_to_source',
                         )

        self.cached = cached
        self.cached_propagation = None
        self.batch_size_messages = batch_size_messages

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Tensor = None,
                ) -> Tensor:
        if self.cached_propagation is not None:
            propagation = self.cached_propagation
        else:
            if self.batch_size_messages is not None:
                propagation = torch.zeros(x.shape, device=x.device)
                for batch_start in range(0, edge_index.shape[1], self.batch_size_messages):
                    batch_end = batch_start + self.batch_size_messages
                    if batch_end > edge_index.shape[1]:
                        batch_end = edge_index.shape[1]
                    propagation += self.propagate(edge_index=edge_index[:, batch_start:batch_end],
                                                  x=x,
                                                  edge_weight=edge_weight[batch_start:batch_end],
                                                  size=None,
                                                  )
            else:
                propagation = self.propagate(edge_index=edge_index,
                                             x=x,
                                             edge_weight=edge_weight,
                                             size=None,
                                             )

            if self.cached:
                self.cached_propagation = propagation.detach()

        return propagation

    def message(self,
                x_j: Tensor,
                edge_weight: Tensor,
                ) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def reset_parameters(self):
        self.cached_propagation = None


class PushNet(Module):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 variant: str,
                 dropout: float,
                 bias: bool,
                 hidden_size: int = None,
                 batch_size_messages: Optional[int] = None,
                 ):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.variant = variant
        self.bias = bias

        if variant == 'TPP':
            in_features_predictor = hidden_size
            feature_transform = True
            cached = False
            ppr_dropout = True
        elif variant == 'PP':
            in_features_predictor = in_features
            feature_transform = False
            cached = True
            ppr_dropout = False
        elif variant == 'PTP':
            in_features_predictor = hidden_size
            feature_transform = True
            cached = True
            ppr_dropout = False
        elif variant == 'Full':
            in_features_predictor = hidden_size
            feature_transform = True
            cached = False
            ppr_dropout = True
        else:
            raise ValueError('Unknown PushNet variant: {}'.format(variant))

        if feature_transform:
            self.transformer = Sequential(Dropout(dropout),
                                          Linear(in_features, hidden_size, bias=bias),
                                          ReLU(),
                                          )
        self.propagator = PushNetConv(cached=cached,
                                      batch_size_messages=batch_size_messages,
                                      )
        self.predictor = Sequential(Dropout(dropout),
                                    Linear(in_features_predictor, num_classes, bias=bias),
                                    )
        if ppr_dropout:
            self.ppr_dropout = PropagationDropout(dropout)

        if not feature_transform:
            self.l2_reg_params = list(self.predictor[1].parameters())
        else:
            self.l2_reg_params = list(self.transformer[1].parameters())

        self.reset_parameters()

    def forward(self,
                data: Data,
                ) -> Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        if self.variant == 'TPP':
            x = self.transformer(x)
            x = self.predictor(x)
            edge_index, edge_weight = self.ppr_dropout(edge_index, edge_weight)
            x = self.propagator(x=x,
                                edge_index=edge_index,
                                edge_weight=edge_weight,
                                )
        elif self.variant == 'PP':
            x = self.propagator(x=x,
                                edge_index=edge_index,
                                edge_weight=edge_weight,
                                )
            x = self.predictor(x)
        elif self.variant == 'PTP':
            x = self.propagator(x=x,
                                edge_index=edge_index,
                                edge_weight=edge_weight,
                                )
            x = self.transformer(x)
            x = self.predictor(x)
        elif self.variant == 'Full':
            x = self.transformer(x)
            edge_index, edge_weight = self.ppr_dropout(edge_index, edge_weight)
            x = self.propagator(x=x,
                                edge_index=edge_index,
                                edge_weight=edge_weight,
                                )
            x = self.predictor(x)

        x = log_softmax(x, dim=1)

        return x

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, Linear):
                kaiming_uniform_(module.weight,
                                 a=math.sqrt(5),
                                 )
                if module.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    uniform_(module.bias,
                             a=-bound,
                             b=bound,
                             )
        self.propagator.reset_parameters()

    def __repr__(self):
        return '{}-{}({}, {})'.format(self.__class__.__name__,
                                      self.variant,
                                      self.in_features,
                                      self.num_classes,
                                      )
