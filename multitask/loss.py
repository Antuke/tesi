""" Multitask loss, with masked loss """
from torch import Tensor

class MaskedLoss:
    '''
        Custom masked loss function
    '''
    def forward(self, logits: Tensor, target: Tensor, loss_fn) -> Tensor:
        """
        Computes the masked loss between logits and target tensors.

        Parameters:
        - logits (Tensor): The predicted output tensor.
        - target (Tensor): The ground truth tensor.
        - loss_fn (function): The loss function to be applied.

        Returns:
        - Tensor: The computed masked loss. If the loss function has no reduction than the loss tensor has the size of the batch, so we compute the mean,
        otherwise if the loss function has a reduction applied we return directly the value.
        """
        mask = target != -1
        loss = loss_fn(logits[mask], target[mask])
        if loss.size == 1:
            return loss
        else:
            return loss.mean()

        
    def __call__(self, input: Tensor, target: Tensor, loss_fn) -> Tensor:
        return self.forward(input, target, loss_fn)