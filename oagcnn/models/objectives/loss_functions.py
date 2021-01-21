import torch


torch.manual_seed(0)

def MaskedNLL(target, probs, balance_weights=None):
    # adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, ) which contains the index of the true
            class for each corresponding step.
        probs: A Variable containing a FloatTensor of size
            (batch, num_classes) which contains the
            softmax probability for each class.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    log_probs = torch.log(probs)

    if balance_weights is not None:
        balance_weights = balance_weights.cuda()
        log_probs = torch.mul(log_probs, balance_weights)

    losses = -torch.gather(log_probs, dim=1, index=target)
    return losses.squeeze()

def StableBalancedMaskedBCE(target, out, balance_weight = None):
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    if balance_weight is None:
        num_positive = target.sum()
        num_negative = (1 - target).sum()
        total = num_positive + num_negative
        balance_weight = num_positive / total

    max_val = (-out).clamp(min=0)
    # bce with logits
    loss_values =  out - out * target + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()
    loss_positive = loss_values*target
    loss_negative = loss_values*(1-target)
    losses = (1-balance_weight)*loss_positive + balance_weight*loss_negative

    return losses.squeeze()


def softIoU(target, out, e=1e-6):

    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """

    # (BATCH, NUM_OBJ, H, W) -> (BATCH, NUM_OBJ, 1) -> Loss for each object from each batch

    num = torch.sum(out*target, dim=2)
    den1 = out+target
    den2 = out*target
    den3 = den1 - den2
    den = torch.sum(den3, dim=2) + e
    iou = num / den

    cost = (1 - iou)

    return cost.squeeze()
