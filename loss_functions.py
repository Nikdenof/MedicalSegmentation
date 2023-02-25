import torch

def bce_loss(y_real, y_pred):
    # y_pred = torch.sigmoid(y_pred)#probably could be omitted
    # loss = (torch.max(y_pred,torch.zeros_like(y_pred)) - y_real * y_pred + torch.log(1 + torch.exp(-y_pred))).mean() #tensorflow tutorial variant

    # maybe I should apply softmax, not sure
    # sigmoid is often used instead of softmax in binary classification, here we can simplify it
    loss = (y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred))).mean() #advised variant
    return loss
