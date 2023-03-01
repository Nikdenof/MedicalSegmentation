import torch
#from torch import nn

# Maybe I need to make a class, that the others will inherit from

def binary_cross_ent_loss(y_real, y_predicted):
    # y_predicted = torch.sigmoid(y_predicted)#probably could be omitted
    # loss = (torch.max(y_predicted,torch.zeros_like(y_predicted)) - y_real * y_predicted + torch.log(1 + torch.exp(-y_predicted))).mean() #tensorflow tutorial variant

    # maybe I should apply softmax, not sure
    # sigmoid is often used instead of softmax in binary classification, here we can simplify it
    loss = (y_predicted - y_real * y_predicted + torch.log(1 + torch.exp(-y_predicted))).mean()
    return loss

def dice_loss(y_real, y_predicted, eps = 2e-8):
    y_predicted = torch.sigmoid(y_predicted)
    num = ((2 * y_real * y_predicted).mean())
    den = (y_real + y_predicted).mean() + eps #to avoid deviding by 0
    # res = 1 - (num/den).mean()
    loss = 1 - (num/den)
    return loss 

def focal_loss(y_real, y_predicted, eps = 1e-8, gamma = 2):
    # чтобы не допустить значения логорифма ограничиваем
    # y_predicted = torch.clamp(torch.sigmoid(y_predicted), min=0+eps, max=1-eps) # hint: torch.clamp
    y_predicted = torch.clamp(torch.sigmoid(y_predicted), min=0+eps) # hint: torch.clamp
    #max может и не нужен
    your_loss = -1*((((1 - y_predicted)**gamma)*y_real*torch.log(y_predicted))
                + (1 - y_real)*torch.log((1 - y_predicted))).mean()
    return your_loss
