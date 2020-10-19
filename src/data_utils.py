import os
from typing import Tuple, List, Dict

import networkx
import numpy
import torch
from sklearn.preprocessing import normalize
from torch import Tensor
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_networkx, from_networkx, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add


def get_preprocessed_path(dataset_name: str,
                          data_root: str,
                          sub_folder: str = None,
                          make_dirs: bool = True,
                          ) -> str:
    sub_path = os.path.join(data_root,
                            'preprocessed',
                            dataset_name,
                            )
    if sub_folder is not None:
        sub_path = os.path.join(sub_path, sub_folder)
    if not os.path.exists(sub_path):
        if make_dirs:
            os.makedirs(sub_path)
    return sub_path


def save_adj(edge_index: Tensor,
             edge_weight: Tensor,
             only_largest_cc: bool,
             add_self_loops: bool,
             adj_normalization: str,
             dataset_name: str,
             data_root: str,
             ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     )
    file_name = '{}'.format(dataset_name)
    if only_largest_cc:
        file_name += '_largest_cc'
    if add_self_loops:
        file_name += '_self_loops'
    if adj_normalization is not None:
        file_name += '_{}'.format(adj_normalization)

    file_name_index = file_name + '.edge_index'
    torch.save(edge_index,
               os.path.join(save_dir, file_name_index),
               )

    if edge_weight is not None:
        file_name_weight = file_name + '.edge_weight'
        torch.save(edge_weight,
                   os.path.join(save_dir, file_name_weight),
                   )


def save_ppr(edge_index: Tensor,
             edge_weight: Tensor,
             only_largest_cc: bool,
             add_self_loops: bool,
             adj_normalization: str,
             alpha: float,
             epsilon: float,
             dataset_name: str,
             data_root: str,
             ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='ppr',
                                     )
    file_name = '{}'.format(dataset_name)
    if only_largest_cc:
        file_name += '_largest_cc'
    if add_self_loops:
        file_name += '_self_loops'
    if adj_normalization is not None:
        file_name += '_{}'.format(adj_normalization)
    file_name += '_alpha={:.2e}_epsilon={:.2e}'.format(alpha, epsilon)
    file_name_edge_index = file_name + '.edge_index'
    file_name_edge_weight = file_name + '.edge_weight'

    torch.save(edge_index,
               os.path.join(save_dir, file_name_edge_index),
               )
    torch.save(edge_weight,
               os.path.join(save_dir, file_name_edge_weight),
               )


def save_features(x: Tensor,
                  normalization: str,
                  dataset_name: str,
                  data_root: str,
                  ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     )
    file_name = '{}'.format(dataset_name)
    if normalization is not None:
        file_name += '_{}'.format(normalization)
    file_name += '.x'

    torch.save(x,
               os.path.join(save_dir, file_name)
               )


def save_labels(y: Tensor,
                dataset_name: str,
                data_root: str,
                ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     )
    file_name = '{}.y'.format(dataset_name)

    torch.save(y,
               os.path.join(save_dir, file_name)
               )


def save_split(train_mask: Tensor,
               val_mask: Tensor,
               test_mask: Tensor,
               dataset_name: str,
               data_root: str,
               ) -> None:
    save_dir = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     )
    torch.save(train_mask,
               os.path.join(save_dir,
                            '{}.train_mask'.format(dataset_name),
                            )
               )
    torch.save(val_mask,
               os.path.join(save_dir,
                            '{}.val_mask'.format(dataset_name),
                            )
               )
    torch.save(test_mask,
               os.path.join(save_dir,
                            '{}.test_mask'.format(dataset_name),
                            )
               )


def load_adj(only_largest_cc: bool,
             add_self_loops: bool,
             adj_normalization: str,
             dataset_name: str,
             data_root: str,
             ) -> Tuple[Tensor, Tensor]:
    preprocessed_path = get_preprocessed_path(dataset_name=dataset_name,
                                              data_root=data_root,
                                              make_dirs=False,
                                              )
    file_name = '{}'.format(dataset_name)
    if only_largest_cc:
        file_name += '_largest_cc'
    if add_self_loops:
        file_name += '_self_loops'
    if adj_normalization is not None:
        file_name += '_{}'.format(adj_normalization)

    file_name_index = file_name + '.edge_index'
    edge_index = torch.load(os.path.join(preprocessed_path, file_name_index))

    try:
        file_name_weight = file_name + '.edge_weight'
        edge_weight = torch.load(os.path.join(preprocessed_path, file_name_weight))
    except FileNotFoundError:
        edge_weight = None

    return edge_index, edge_weight


def load_ppr(only_largest_cc: bool,
             add_self_loops: bool,
             adj_normalization: str,
             alpha: float,
             epsilon: float,
             dataset_name: str,
             data_root: str,
             ) -> Tuple[Tensor, Tensor]:
    ppr_path = get_preprocessed_path(dataset_name=dataset_name,
                                     data_root=data_root,
                                     sub_folder='ppr',
                                     make_dirs=False,
                                     )
    file_name = '{}'.format(dataset_name)
    if only_largest_cc:
        file_name += '_largest_cc'
    if add_self_loops:
        file_name += '_self_loops'
    if adj_normalization is not None:
        file_name += '_{}'.format(adj_normalization)
    file_name += '_alpha={:.2e}_epsilon={:.2e}'.format(alpha, epsilon)
    file_name_edge_index = file_name + '.edge_index'
    file_name_edge_weight = file_name + '.edge_weight'

    edge_index = torch.load(os.path.join(ppr_path, file_name_edge_index))
    edge_weight = torch.load(os.path.join(ppr_path, file_name_edge_weight))

    return edge_index, edge_weight


def load_features(normalization: str,
                  dataset_name: str,
                  data_root: str,
                  ) -> Tensor:
    preprocessed_path = get_preprocessed_path(dataset_name=dataset_name,
                                              data_root=data_root,
                                              make_dirs=False,
                                              )
    file_name = '{}'.format(dataset_name)
    if normalization is not None:
        file_name += '_{}'.format(normalization)
    file_name += '.x'

    x = torch.load(os.path.join(preprocessed_path, file_name))
    return x


def load_labels(dataset_name: str,
                data_root: str,
                ) -> Tensor:
    preprocessed_path = get_preprocessed_path(dataset_name=dataset_name,
                                              data_root=data_root,
                                              make_dirs=False,
                                              )
    file_name = '{}.y'.format(dataset_name)

    y = torch.load(os.path.join(preprocessed_path, file_name))
    return y


def load_split(dataset_name: str,
               data_root: str,
               ) -> Tuple[Tensor, Tensor, Tensor]:
    preprocessed_path = get_preprocessed_path(dataset_name=dataset_name,
                                              data_root=data_root,
                                              make_dirs=False,
                                              )
    train_mask = torch.load(os.path.join(preprocessed_path,
                                         '{}.train_mask'.format(dataset_name),
                                         ),
                            )
    val_mask = torch.load(os.path.join(preprocessed_path,
                                       '{}.val_mask'.format(dataset_name),
                                       ),
                          )
    test_mask = torch.load(os.path.join(preprocessed_path,
                                        '{}.test_mask'.format(dataset_name),
                                        ),
                           )

    return train_mask, val_mask, test_mask


def load_dataset(only_largest_cc: bool,
                 add_self_loops: bool,
                 adj_normalization: str,
                 x_normalization: str,
                 dataset_name: str,
                 data_root: str,
                 alpha: float = None,
                 epsilon: float = None,
                 ) -> Data:
    if alpha is None or epsilon is None:
        # Load adjacency matrix
        edge_index = load_adj(only_largest_cc=only_largest_cc,
                              add_self_loops=add_self_loops,
                              adj_normalization=adj_normalization,
                              dataset_name=dataset_name,
                              data_root=data_root,
                              )
        edge_weight = None
    else:
        # Load ppr matrix
        edge_index, edge_weight = load_ppr(only_largest_cc=only_largest_cc,
                                           add_self_loops=add_self_loops,
                                           adj_normalization=adj_normalization,
                                           alpha=alpha,
                                           epsilon=epsilon,
                                           dataset_name=dataset_name,
                                           data_root=data_root,
                                           )

    # Load feature matrix
    x = load_features(normalization=x_normalization,
                      dataset_name=dataset_name,
                      data_root=data_root,
                      )

    # Load label vector
    y = load_labels(dataset_name=dataset_name,
                    data_root=data_root, )

    # Create dataset
    data = Data(edge_index=edge_index,
                edge_attr=edge_weight,
                x=x,
                y=y,
                )

    return data


def preprocess_dataset(dataset: Dataset,
                       only_largest_cc: bool,
                       add_self_loops: bool,
                       adj_normalization: str,
                       x_normalization: str,
                       ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Get data
    data = dataset[0]
    edge_index = data.edge_index
    x = dataset.data.x.numpy().astype(np.float32)
    y = dataset.data.y

    # Restrict graph to largest connected component
    if only_largest_cc:
        graph = to_networkx(data,
                            to_undirected=False,
                            )
        largest_cc = max(networkx.weakly_connected_components(graph), key=len)
        nodes_cc = numpy.sort(list(largest_cc))
        graph = graph.subgraph(largest_cc)
        data = from_networkx(graph)

        edge_index = data.edge_index
        x = x[nodes_cc, :]
        y = y[nodes_cc]

    # Add self-loops
    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index,
                                                           num_nodes=data.num_nodes,
                                                           )
    else:
        edge_weight = None

    # Normalize adjacency matrix
    edge_index, edge_weight = normalize_adj(edge_index=edge_index,
                                            edge_weight=edge_weight,
                                            num_nodes=data.num_nodes,
                                            dtype=dataset.data.x.dtype,
                                            normalization=adj_normalization,
                                            )

    # Normalize features
    x = normalize(x,
                  norm=x_normalization,
                  axis=1,
                  )
    x = torch.FloatTensor(x)

    return edge_index, edge_weight, x, y


def normalize_adj(edge_index: Tensor,
                  edge_weight: Tensor = None,
                  num_nodes: int = None,
                  dtype: torch.dtype = None,
                  normalization: str = 'sym',
                  ) -> Tuple[Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.shape[1],),
                                 dtype=dtype,
                                 device=edge_index.device,
                                 )

    if normalization is None:
        return edge_index, edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg = deg.pow_(-0.5)
        deg.masked_fill_(deg == float('inf'), 0)
        edge_weight = deg[row] * edge_weight * deg[col]
    elif normalization == 'rw':
        deg = deg.pow_(-1.)
        deg.masked_fill_(deg == float('inf'), 0)
        edge_weight = edge_weight * deg[col]
    else:
        raise ValueError('Unknown normalization: {}'.format(normalization))

    return edge_index, edge_weight


def get_ppr_matrix_dense(adj: numpy.ndarray,
                         alpha: float,
                         epsilon: float,
                         ) -> numpy.ndarray:
    # Compute PPR matrix
    n = adj.shape[0]
    ppr = alpha * numpy.linalg.inv(numpy.eye(n) - (1. - alpha) * adj)

    # Sparsify
    ppr[ppr < epsilon] = 0.

    # L1-normalize rows
    ppr = normalize(ppr,
                    norm='l1',
                    axis=1,
                    )

    return ppr


def add_sparse(edge_index: List[Tensor],
               edge_weight: List[Tensor],
               num_nodes: int = None,
               ) -> Tuple[Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index[0], num_nodes)

    sparse_adj = [torch.sparse.FloatTensor(edge_index_alpha, edge_weight_alpha, (num_nodes, num_nodes)) for
                  edge_index_alpha, edge_weight_alpha in zip(edge_index, edge_weight)]
    adj_sum = torch.sparse.sum(torch.stack(sparse_adj, dim=0), dim=0)

    edge_index, edge_weight = adj_sum._indices(), adj_sum._values()

    return edge_index, edge_weight


def index_to_mask(index: Tensor,
                  size: int,
                  ) -> Tensor:
    mask = torch.zeros(size,
                       dtype=torch.bool,
                       device=index.device,
                       )
    mask[index] = 1

    return mask


def get_random_data_split(data: Data,
                          num_classes: int,
                          num_train_per_class: int = 20,
                          num_val: int = 500,
                          ) -> Dict[str, Tensor]:
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:num_train_per_class] for i in indices], dim=0)

    rest_index = torch.cat([i[num_train_per_class:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=data.num_nodes)
    val_mask = index_to_mask(rest_index[:num_val], size=data.num_nodes)
    test_mask = index_to_mask(rest_index[num_val:], size=data.num_nodes)

    split_dict = {'train_mask': train_mask,
                  'val_mask': val_mask,
                  'test_mask': test_mask,
                  }

    return split_dict
