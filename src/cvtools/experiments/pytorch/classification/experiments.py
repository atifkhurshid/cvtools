"""
Script for running experiments.
"""

# Author: Atif Khurshid
# Created: 2026-05-17
# Modified: 2026-10-28
# Version: 1.4
# Changelog:
#     - 2026-05-17: Initial version.
#     - 2026-05-19: Added print statements.
#     - 2026-05-19: Added cleanup to free memory after each experiment.
#     - 2026-05-20: Added support for wandb groups.
#     - 2026-05-26: Added error handling.
#     - 2026-10-28: Updated to use new dataset and model config classes.

from typing import Optional
from datetime import datetime

import os
import gc
import time
import wandb
import pandas as pd

from torchview import draw_graph
from cvtools.models.pytorch.utils import train_classification_model
from cvtools.models.pytorch.utils import test_classification_model

from ...configs.pytorch import PyTorchClassificationDatasetConfig
from ...configs.pytorch import PyTorchClassificationModelConfig
from ._evaluation import evaluate_model, evaluate_multilabel_model


def run_experiment(
        name: str,
        dataset_config: PyTorchClassificationDatasetConfig,
        model_config: PyTorchClassificationModelConfig,
        wandb_config: Optional[dict] = None,
        max_confusion_classes: int = 20,
        max_roc_classes: int = 16,
    ) -> dict:
    """
    Run a single experiment with the given dataset and model configurations.

    Parameters
    ----------
    name : str
        Name of the experiment.
    dataset_config : PyTorchClassificationDatasetConfig
        Configuration for the dataset.
    model_config : PyTorchClassificationModelConfig
        Configuration for the model.
    wandb_config : dict, optional
        Configuration for Weights & Biases logging. If None, logging is disabled.
    max_confusion_classes : int, optional
        Maximum number of classes to log confusion matrix for. If the number of classes exceeds this,
        the confusion matrix will not be logged to Weights & Biases.
    max_roc_classes : int, optional
        Maximum number of classes to log individual ROC curves for. If the number of classes exceeds this,
        only micro, macro, and weighted ROC curves will be logged.
    """
    run = None

    try:
        print("Loading data...", end="", flush=True)
        dataloader_train, dataloader_val, dataloader_test = dataset_config.load_data()
        print("Done")

        model_config.n_classes = dataset_config.n_classes
        model_config.model_kwargs["n_classes"] = dataset_config.n_classes
        model_config.class_weights = dataset_config.class_weights

        print("Preparing model...", end="", flush=True)
        model = model_config.prepare_model()
        print("Done")

        if wandb_config is not None:
            print("Initializing Weights & Biases run...", flush=True)
            config = {
                "dataset": dataset_config,
                "model": model_config,
            }
            run = wandb.init(
                entity = wandb_config["entity"],
                project = wandb_config["project"],
                dir = wandb_config["dir"],
                id = name + "-" + datetime.now().strftime("%Y%m%d%H%M%S"),
                name = name,
                resume = wandb_config.get("resume", "never"),
                config = config,
                group = wandb_config.get("group", None),
                reinit = "finish_previous",
            )

            print("Logging model architecture...", end="", flush=True)
            model_graph_filepath = os.path.join(wandb_config["temp_dir"], "model_graph")
            model_graph = draw_graph(model, input_size = model_config.input_shape)
            model_graph.visual_graph.render(model_graph_filepath, format="png", cleanup=True)
            run.log({
                "model": wandb.Image(model_graph_filepath + ".png", caption="Model Architecture")
            })
            print("Done", flush=True)

        pretrain = getattr(model_config, "pretrain", False)
        pretrain_epochs = getattr(model_config, "pretrain_epochs", 0)
        starting_epoch = 0
        if pretrain and pretrain_epochs > 0:
            print("Pretraining classifier with frozen backbone...", flush=True)
            if hasattr(model, "freeze_backbone") and hasattr(model, "unfreeze_backbone"):
                model.freeze_backbone()
                train_classification_model(
                    model = model,
                    train_dataloader = dataloader_train,
                    epochs = pretrain_epochs,
                    val_dataloader = dataloader_val,
                    val_strategy = model_config.val_strategy,
                    early_stopping = False,
                    restore_best_weights = False,
                    run = run,
                    log_interval = dataset_config.train_batches // model_config.log_frequency,
                    verbose = True,
                    pbar = False,
                )
                model.unfreeze_backbone()
                starting_epoch = pretrain_epochs
                model.set_learning_rate(model_config.learning_rate * 0.1) # Reduce learning rate after pretraining
            else:
                print("Warning: Model does not have a freezable backbone. Skipping pretraining.", flush=True)

        print("Training model...", flush=True)
        train_classification_model(
            model = model,
            train_dataloader = dataloader_train,
            epochs = model_config.epochs - starting_epoch,
            starting_epoch = starting_epoch,
            val_dataloader = dataloader_val,
            val_strategy = model_config.val_strategy,
            early_stopping = model_config.early_stopping,
            patience = model_config.early_stopping_patience,
            min_delta = model_config.early_stopping_min_delta,
            restore_best_weights = model_config.early_stopping_restore_best_weights,
            run = run,
            log_interval = dataset_config.train_batches // model_config.log_frequency,
            checkpoint_dir = model_config.checkpoint_dir,
            checkpoint_interval = model_config.checkpoint_interval,
            verbose = True,
            pbar = False,
        )

        if dataloader_test is not None:
            print("Testing model...", flush=True)
            _, _, outputs, y_pred, y_true = test_classification_model(
                model = model,
                data = dataloader_test,
                return_outputs = True,
                pbar = False,
            )

            if model_config.multilabel:
                summary = evaluate_multilabel_model(
                    y_true = y_true,
                    y_pred = y_pred,
                    outputs = outputs,
                    classes = dataset_config.classes,
                    run = run,
                )
            else:
                summary = evaluate_model(
                    y_true = y_true,
                    y_pred = y_pred,
                    outputs = outputs,
                    classes = dataset_config.classes,
                    max_confusion_classes = max_confusion_classes,
                    max_roc_classes = max_roc_classes,
                    run = run,
                )

        else:
            summary = {}

        if run is not None:
            print("Finishing Weights & Biases run...")
            run.finish()
            run = None
    
    except Exception as e:
        print(f"\nError occurred while running experiment '{name}':\n{e}")
        summary = {}

    finally:

        if run is not None:
            print("Finishing Weights & Biases run...")
            run.finish()

        print("Cleaning up...", end="", flush=True)

        _del(model)
        _del(dataloader_train.dataset)
        _del(dataloader_train._iterator)

        if dataloader_val is not None:
            _del(dataloader_val.dataset)
            _del(dataloader_val._iterator)

        if dataloader_test is not None:
            _del(dataloader_test.dataset)
            _del(dataloader_test._iterator)

        _del(dataloader_train)
        _del(dataloader_val)
        _del(dataloader_test)

        gc.collect()

        time.sleep(5) # Sleep to ensure resources are released before next experiment

        print("Done")

        return summary


def run_experiment_with_trials(
        name: str,
        dataset_config: PyTorchClassificationDatasetConfig,
        model_config: PyTorchClassificationModelConfig,
        wandb_config: Optional[dict] = None,
        trials: int = 5,
        max_confusion_classes: int = 20,
        max_roc_classes: int = 16,
    ):
    """
    Run an experiment multiple times with different random seeds.

    Parameters
    ----------
    name : str
        Name of the experiment.
    dataset_config : PyTorchClassificationDatasetConfig
        Configuration for the dataset.
    model_config : PyTorchClassificationModelConfig
        Configuration for the model.
    wandb_config : dict, optional
        Configuration for Weights & Biases logging. If None, logging is disabled.
    trials : int, optional
        Number of trials to run. Each trial will use a different random seed.
    max_confusion_classes : int, optional
        Maximum number of classes to include in the confusion matrix.
    max_roc_classes : int, optional
        Maximum number of classes to include in the ROC curve.
    """
    if wandb_config is not None:
        wandb_config["group"] = wandb_config.get(
            "group",    # Use existing group if provided
            name + "-" + datetime.now().strftime("%Y%m%d%H%M"), # Else group by experiment name
        )

    trial_summaries = []
    for trial in range(trials):
        trial_name = f"{name}-trial{trial+1}"
        print(f"Running trial {trial+1}/{trials}...\n")

        summary = run_experiment(
            name = trial_name,
            dataset_config = dataset_config,
            model_config = model_config,
            wandb_config = wandb_config,
            max_confusion_classes = max_confusion_classes,
            max_roc_classes = max_roc_classes,
        )

        trial_summaries.append(summary)
    
    df = pd.DataFrame(trial_summaries)
    df = df.rename_axis("Trial")
    df.loc["mean"] = df.mean()
    df.loc["std"] = df.std()

    print("\nTrial Summary:\n")
    for metric in df.columns:
        print(f"\t{metric}: ", end="")
        print(f"{df.loc['mean', metric]:.4f} ± {df.loc['std', metric]:.4f}")
    print()

    if wandb_config is not None:
        print("Initializing Weights & Biases run...")
        name = name + "-trialsummary"
        run = wandb.init(
            entity = wandb_config["entity"],
            project = wandb_config["project"],
            dir = wandb_config["dir"],
            id = name + "-" + datetime.now().strftime("%Y%m%d%H%M%S"),
            name = name,
            resume = wandb_config.get("resume", "never"),
            group = wandb_config.get("group", None),
            reinit = "finish_previous",
        )
        
        run.log({
            "trials_summary": wandb.Table(dataframe=df)
        })

        print("Finishing Weights & Biases run...")
        run.finish()


def _del(obj):
    try:
        del obj
    except:
        pass
