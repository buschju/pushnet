import time
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import nll_loss
from torch.optim import Adam, Optimizer
from torch_geometric.data import Data


def train_step(model: Module,
               data: Data,
               optimizer: Optimizer,
               l2_reg: float,
               ) -> None:
    model.train()

    optimizer.zero_grad()
    log_probabilities = model(data)
    loss = nll_loss(log_probabilities[data.train_mask],
                    data.y[data.train_mask],
                    )
    l2_reg_loss = sum(((param ** 2).sum() for param in model.l2_reg_params)) / 2.
    loss += l2_reg * l2_reg_loss
    loss.backward()
    optimizer.step()


def evaluate_model(model: Module,
                   data: Data,
                   l2_reg: float,
                   ) -> Dict[str, float]:
    model.eval()

    with torch.no_grad():
        log_probabilities = model(data)

    results = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]

        loss = nll_loss(log_probabilities[mask], data.y[mask]).item()
        l2_reg_loss = sum(((param ** 2).sum().item() for param in model.l2_reg_params)) / 2.
        loss += l2_reg * l2_reg_loss

        pred = log_probabilities[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        results['{}_loss'.format(key)] = loss
        results['{}_acc'.format(key)] = acc

    return results


def train_random_splits(splits: List[Dict[str, Tensor]],
                        num_runs_per_split: int,
                        model: Module,
                        data: Data,
                        learning_rate: float,
                        patience: int,
                        max_epochs: int,
                        l2_reg: float,
                        ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    num_random_splits = len(splits)

    train_accs = []
    val_accs = []
    test_accs = []
    best_epochs = []
    seconds_per_epoch = []

    for i in range(num_random_splits):
        data.train_mask = splits[i]['train_mask']
        data.val_mask = splits[i]['val_mask']
        data.test_mask = splits[i]['test_mask']

        # Get results from multiple runs
        train_accs_split, val_accs_split, test_accs_split, best_epochs_split, seconds_per_epoch_split = train_single_split(
            num_runs=num_runs_per_split,
            model=model,
            data=data,
            learning_rate=learning_rate,
            patience=patience,
            max_epochs=max_epochs,
            l2_reg=l2_reg,
        )

        train_accs += train_accs_split
        val_accs += val_accs_split
        test_accs += test_accs_split
        best_epochs += best_epochs_split
        seconds_per_epoch += seconds_per_epoch_split

    return train_accs, val_accs, test_accs, best_epochs, seconds_per_epoch


def train_single_split(num_runs: int,
                       model: Module,
                       data: Data,
                       learning_rate: float,
                       patience: int,
                       max_epochs: int,
                       l2_reg: float,
                       ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    train_accs = []
    val_accs = []
    test_accs = []
    best_epochs = []
    seconds_per_epoch = []

    for _ in range(num_runs):
        # Reset model parameters
        model.reset_parameters()

        # Single run
        train_acc_run, val_acc_run, test_acc_run, best_epoch_run, seconds_per_epoch_run = train_single_run(model=model,
                                                                                                           data=data,
                                                                                                           learning_rate=learning_rate,
                                                                                                           patience=patience,
                                                                                                           max_epochs=max_epochs,
                                                                                                           l2_reg=l2_reg,
                                                                                                           )
        train_accs += [train_acc_run]
        val_accs += [val_acc_run]
        test_accs += [test_acc_run]
        best_epochs += [best_epoch_run]
        seconds_per_epoch += [seconds_per_epoch_run]

    return train_accs, val_accs, test_accs, best_epochs, seconds_per_epoch


def train_single_run(model: Module,
                     data: Data,
                     learning_rate: float,
                     patience: int,
                     max_epochs: int,
                     l2_reg: float,
                     ) -> Tuple[float, float, float, float, float]:
    optimizer = Adam(params=model.parameters(),
                     lr=learning_rate,
                     )

    best_results = None
    best_val_acc = 0
    best_val_loss = float('inf')
    best_epoch = -1
    bad_counter = 0

    num_epochs = None
    start_time = time.time()

    for epoch in range(1, max_epochs - 1):
        # Train step
        train_step(model=model,
                   data=data,
                   optimizer=optimizer,
                   l2_reg=l2_reg,
                   )

        # Evaluate on validation set
        evaluation_results = evaluate_model(model=model,
                                            data=data,
                                            l2_reg=l2_reg,
                                            )
        val_acc_epoch = evaluation_results['val_acc']
        val_loss_epoch = evaluation_results['val_loss']

        # Early stopping
        if val_acc_epoch >= best_val_acc or val_loss_epoch <= best_val_loss:
            if val_acc_epoch >= best_val_acc and val_loss_epoch <= best_val_loss:
                best_results = evaluation_results
                best_epoch = epoch
            best_val_acc = max(val_acc_epoch, best_val_acc)
            best_val_loss = min(val_loss_epoch, best_val_loss)
            bad_counter = 0
        else:
            bad_counter += 1
            if bad_counter == patience:
                num_epochs = epoch
                break

    end_time = time.time()
    if num_epochs is None:
        num_epochs = max_epochs
    seconds_per_epoch = (end_time - start_time) / num_epochs

    return best_results['train_acc'], best_results['val_acc'], best_results['test_acc'], best_epoch, seconds_per_epoch
