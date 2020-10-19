import argparse
import json
import random

import numpy
import torch
from torch_geometric.data import Data

import mlflow
from data_utils import load_ppr, add_sparse, load_adj, load_features, load_labels, get_random_data_split, load_split
from models import get_model_class
from training import train_random_splits, train_single_split
from utils import flatten_dictionary


def run_experiment(data_root: str,
                   dataset_name: str,
                   dataset_parameters: dict,
                   model_name: str,
                   model_parameters: dict,
                   training_parameters: dict,
                   run_parameters: dict,
                   seed: int,
                   device: int,
                   ) -> None:
    # Load propagation matrix
    if 'alphas' in dataset_parameters and 'epsilon' in dataset_parameters:
        # PPR matrix
        edge_index = []
        edge_weight = []
        for alpha in dataset_parameters['alphas']:
            edge_index_alpha, edge_weight_alpha = load_ppr(only_largest_cc=dataset_parameters['only_largest_cc'],
                                                           add_self_loops=dataset_parameters['add_self_loops'],
                                                           adj_normalization=dataset_parameters['adj_normalization'],
                                                           alpha=alpha,
                                                           epsilon=dataset_parameters['epsilon'],
                                                           dataset_name=dataset_name,
                                                           data_root=data_root,
                                                           )
            edge_index += [edge_index_alpha]
            edge_weight += [edge_weight_alpha]
        if len(edge_index) > 1:
            # Perform sum-aggregation
            edge_index, edge_weight = add_sparse(edge_index=edge_index,
                                                 edge_weight=edge_weight,
                                                 )
        else:
            edge_index, edge_weight = edge_index[0], edge_weight[0]
    else:
        # Adjacency matrix
        edge_index, edge_weight = load_adj(only_largest_cc=dataset_parameters['only_largest_cc'],
                                           add_self_loops=dataset_parameters['add_self_loops'],
                                           adj_normalization=dataset_parameters['adj_normalization'],
                                           dataset_name=dataset_name,
                                           data_root=data_root,
                                           )

    # Load features and labels
    x = load_features(normalization=dataset_parameters['x_normalization'],
                      dataset_name=dataset_name,
                      data_root=data_root,
                      )
    y = load_labels(dataset_name=dataset_name,
                    data_root=data_root,
                    )
    in_features = x.shape[1]
    num_classes = torch.unique(y).numel()

    # Data
    data = Data(edge_index=edge_index,
                edge_attr=edge_weight,
                x=x,
                y=y,
                )

    # Fix random seed
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get random data splits
    if 'fixed_split' not in run_parameters:
        splits = [get_random_data_split(data=data,
                                        num_classes=num_classes,
                                        )
                  for _ in range(run_parameters['num_random_splits'])
                  ]

    # Model
    model = get_model_class(model_name=model_name)(**model_parameters,
                                                   in_features=in_features,
                                                   num_classes=num_classes,
                                                   )

    # Move to device
    if device is not None:
        device = torch.device(device)
        data = data.to(device)
        model = model.to(device)

    # Run
    if 'fixed_split' in run_parameters:
        # Load pre-defined split
        train_mask, val_mask, test_mask = load_split(dataset_name=dataset_name,
                                                     data_root=data_root,
                                                     )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        # Run on single fixed split
        train_accs, val_accs, test_accs, best_epochs, seconds_per_epoch = train_single_split(
            num_runs=run_parameters['num_runs_per_split'],
            model=model,
            data=data,
            learning_rate=training_parameters[
                'learning_rate'],
            patience=training_parameters['patience'],
            max_epochs=training_parameters['max_epochs'],
            l2_reg=training_parameters['l2_reg'],
        )
    else:
        # Run on multiple generated random splits
        train_accs, val_accs, test_accs, best_epochs, seconds_per_epoch = train_random_splits(splits=splits,
                                                                                              num_runs_per_split=
                                                                                              run_parameters[
                                                                                                  'num_runs_per_split'],
                                                                                              model=model,
                                                                                              data=data,
                                                                                              learning_rate=
                                                                                              training_parameters[
                                                                                                  'learning_rate'],
                                                                                              patience=
                                                                                              training_parameters[
                                                                                                  'patience'],
                                                                                              max_epochs=
                                                                                              training_parameters[
                                                                                                  'max_epochs'],
                                                                                              l2_reg=
                                                                                              training_parameters[
                                                                                                  'l2_reg'],
                                                                                              )

    # Log to MLflow
    for run_idx in range(len(train_accs)):
        mlflow.log_metric('train_acc', train_accs[run_idx], step=run_idx)
        mlflow.log_metric('val_acc', val_accs[run_idx], step=run_idx)
        mlflow.log_metric('test_acc', test_accs[run_idx], step=run_idx)
        mlflow.log_metric('best_epoch', best_epochs[run_idx], step=run_idx)
        mlflow.log_metric('seconds_per_epoch', seconds_per_epoch[run_idx], step=run_idx)
    train_acc_mean = numpy.mean(train_accs)
    train_acc_std = numpy.std(train_accs)
    val_acc_mean = numpy.mean(val_accs)
    val_acc_std = numpy.std(val_accs)
    test_acc_mean = numpy.mean(test_accs)
    test_acc_std = numpy.std(test_accs)
    seconds_per_epoch_mean = numpy.mean(seconds_per_epoch)
    seconds_per_epoch_std = numpy.std(seconds_per_epoch)
    mlflow.log_metric('train_acc_mean', train_acc_mean)
    mlflow.log_metric('train_acc_std', train_acc_std)
    mlflow.log_metric('val_acc_mean', val_acc_mean)
    mlflow.log_metric('val_acc_std', val_acc_std)
    mlflow.log_metric('test_acc_mean', test_acc_mean)
    mlflow.log_metric('test_acc_std', test_acc_std)
    mlflow.log_metric('seconds_per_epoch_mean', seconds_per_epoch_mean)
    mlflow.log_metric('seconds_per_epoch_std', seconds_per_epoch_std)


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        help="Path to config file",
                        type=str,
                        default='../config',
                        )
    parser.add_argument('--data_root',
                        help="Path to data",
                        type=str,
                        default='../data',
                        )
    parser.add_argument('--device',
                        help="Device index",
                        type=int,
                        default=0,
                        )
    parser.add_argument('--mlflow_uri',
                        help="MLflow tracking URI",
                        type=str,
                        default='../mlflow',
                        )
    parser.add_argument('--mlflow_experiment_name',
                        help="Experiment name used for MLflow results tracking",
                        type=str,
                        default='pushnet',
                        )
    args = parser.parse_args()

    # Parse config file
    with open(args.config_path, 'r') as config_file:
        config = json.load(config_file)

    # Set up MLflow results tracking
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(name=args.mlflow_experiment_name)
    if experiment is None:
        mlflow.create_experiment(args.mlflow_experiment_name)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run():
        # Log params
        params_flat = flatten_dictionary(config)
        mlflow.log_params(params_flat)

        # Run experiment
        run_experiment(data_root=args.data_root,
                       dataset_name=config['dataset_name'],
                       dataset_parameters=config['dataset_parameters'],
                       model_name=config['model_name'],
                       model_parameters=config['model_parameters'],
                       training_parameters=config['training_parameters'],
                       run_parameters=config['run_parameters'],
                       seed=config['seed'],
                       device=args.device,
                       )
