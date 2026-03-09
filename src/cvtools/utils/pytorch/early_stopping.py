"""
Early stopping utility for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2026-03-04
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-03-04: Initial implementation of EarlyStopping class.

from torch.nn import Module


class EarlyStopping(object):

    def __init__(
            self,
            patience: int = 5,
            min_delta: float = 1e-4,
            restore_best_weights: bool = True,
        ):
        """
        Implements early stopping to prevent overfitting during training.
        It monitors the validation loss and stops training if it doesn't improve
        for a specified number of epochs (patience).
        Optionally, it can restore the model weights from the epoch
        with the best validation loss.

        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait for an improvement in validation loss before stopping.
        min_delta : float, optional
            Minimum change in validation loss to qualify as an improvement.
        restore_best_weights : bool, optional
            Whether to restore model weights from the epoch with the best validation
            loss after stopping.
        
        Attributes
        ----------
        counter : int
            Counts the number of epochs since the last improvement in validation loss.
        best_loss : float
            The best validation loss observed so far.
        best_state : dict or None
            The model state corresponding to the best validation loss,
            if restore_best_weights is True.
        early_stop : bool
            Indicates whether early stopping has been triggered.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.epochs = 0
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None
        self.best_epoch = 0
        self.early_stop = False


    def step(
            self,
            model: Module,
            val_loss: float,
        ) -> None:
        """
        Checks if the validation loss has improved and updates the internal state
        accordingly. If the validation loss has not improved for a number of epochs
        equal to patience, it sets the early_stop flag to True.

        Parameters
        ----------
        model : Module
            The model being trained, used to save the best weights if 
            restore_best_weights is True.
        val_loss : float
            The validation loss for the current epoch.
        """
        self.epochs += 1
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = self.epochs
            if self.restore_best_weights:
                self.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    

    def restore(self, model: Module) -> None:
        """
        Restores the model weights from the epoch with the best validation loss,
        if restore_best_weights is True and best_state is not None.

        Parameters
        ----------
        model : Module
            The model whose weights are to be restored.
        """
        if self.restore_best_weights and self.best_state is not None and self.counter > 0:
            print("\nRestoring model weights from epoch {} with validation loss {:.4f}".format(
                self.best_epoch, self.best_loss
            ))
            model.load_state_dict(self.best_state)
