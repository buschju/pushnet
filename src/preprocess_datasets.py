import argparse
import os
from itertools import product

from torch import FloatTensor
from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from data_utils import preprocess_dataset, get_ppr_matrix_dense, save_adj, save_features, \
    save_labels, save_ppr

if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        help="Path to data",
                        type=str,
                        default='../data',
                        )
    args = parser.parse_args()
    data_root = args.data_root

    # Set up folder
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    # Constants
    dataset_names = [
        'Citeseer',
    ]
    only_largest_cc = True
    add_self_loops = True
    adj_normalization = 'sym'
    alphas = [
        5.00e-2,
        1.00e-1,
        2.00e-1,
    ]
    epsilons = [
        1.00e-5,
    ]
    x_normalization = 'l1'

    # Preprocess all datasets
    for dataset_name in dataset_names:
        # Load dataset
        if dataset_name in ['Citeseer', 'Cora', 'Pubmed']:
            dataset = Planetoid(root=data_root,
                                name=dataset_name,
                                transform=None,
                                pre_transform=None,
                                )
        elif dataset_name in ['CS', 'Physics']:
            dataset = Coauthor(root=os.path.join(data_root, dataset_name),
                               name=dataset_name,
                               transform=None,
                               pre_transform=None,
                               )

        # Preprocess dataset
        edge_index, edge_weight, x, y = preprocess_dataset(dataset=dataset,
                                                           only_largest_cc=only_largest_cc,
                                                           add_self_loops=add_self_loops,
                                                           adj_normalization=adj_normalization,
                                                           x_normalization=x_normalization,
                                                           )
        if (edge_weight == 1.).all():
            edge_weight = None

        # Save pre-processed dataset
        save_adj(edge_index=edge_index,
                 edge_weight=edge_weight,
                 only_largest_cc=only_largest_cc,
                 add_self_loops=add_self_loops,
                 adj_normalization=adj_normalization,
                 dataset_name=dataset_name,
                 data_root=data_root,
                 )
        save_features(x=x,
                      normalization=x_normalization,
                      dataset_name=dataset_name,
                      data_root=data_root,
                      )
        save_labels(y=y,
                    dataset_name=dataset_name,
                    data_root=data_root,
                    )

        # Compute and save PPR-matrices
        adj = to_dense_adj(edge_index=edge_index,
                           edge_attr=edge_weight,
                           )[0].numpy()
        for alpha, epsilon in product(alphas, epsilons):
            ppr = get_ppr_matrix_dense(adj=adj,
                                       alpha=alpha,
                                       epsilon=epsilon,
                                       )
            edge_index, edge_weight = dense_to_sparse(FloatTensor(ppr))
            save_ppr(edge_index=edge_index,
                     edge_weight=edge_weight,
                     only_largest_cc=only_largest_cc,
                     add_self_loops=add_self_loops,
                     adj_normalization=adj_normalization,
                     alpha=alpha,
                     epsilon=epsilon,
                     dataset_name=dataset_name,
                     data_root=data_root,
                     )
