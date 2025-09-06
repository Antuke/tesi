import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss that supports weights and an ignore index.
    This loss is based on the idea of reformulating ordinal regression as a
    series of binary classification problems.
    """
    def __init__(self, num_classes=9, weight=None, ignore_index=-100):
        """
        Initializes the OrdinalRegressionLoss.

        Args:
            num_classes (int): The total number of ordered classes.
            weight (torch.Tensor, optional): A manual rescaling weight given to each
                                             class. If given, it has to be a Tensor
                                             of size num_classes. Defaults to None.
            ignore_index (int, optional): Specifies a target value that is ignored
                                          and does not contribute to the input gradient.
                                          Defaults to -100.
        """
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # The core of this loss is a binary cross-entropy with logits loss.
        # We set reduction to 'none' to manually handle weighting and masking.
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Register weight as a buffer if it's provided. This ensures it's moved
        # to the correct device along with the module.
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, logits, target):
        """
        Computes the ordinal regression loss.

        Args:
            logits (torch.Tensor): The model's predictions, with shape
                                   (batch_size, num_classes - 1).
            target (torch.Tensor): The ground truth labels, with shape
                                   (batch_size,).

        Returns:
            torch.Tensor: The computed ordinal loss as a single scalar.
        """
        # Create a mask for the targets that should be ignored.
        mask = (target != self.ignore_index)
        
        # If all targets are ignored, return a zero loss.
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Filter out the ignored targets and corresponding logits.
        logits = logits[mask]
        target = target[mask]
        
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Convert the integer target labels to a cumulative binary format.
        # For N classes, we have N-1 binary subproblems.
        # Example: For 5 classes, a target of 2 (0-indexed) becomes [1, 1, 0, 0].
        ordinal_target = torch.zeros_like(logits)
        for i in range(self.num_classes - 1):
            ordinal_target[:, i] = (target > i).float()

        # Compute the binary cross-entropy loss for each subproblem.
        loss = self.bce_loss(logits, ordinal_target)

        # Apply class weights if they are provided.
        if self.weight is not None:
            # Create a weight tensor for the samples in the batch.
            # The weight for each sample is the weight of its class.
            sample_weights = self.weight[target]
            
            # The loss for each sample is the mean of its binary subproblem losses.
            # We then apply the sample weight.
            loss = loss.mean(dim=1) * sample_weights
        
        # The final loss is the mean of the losses for all non-ignored samples.
        return loss.mean()