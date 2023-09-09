from sklearn.linear_model import LinearRegression,LogisticRegression
import pandas as pd
import numpy as np
import math
import torch
from EigenProPytorch import eigenpro
from EigenProPytorch import kernel

def linear_kernel_fn(samples, centers, bandwidth=1000):
    ans = torch.mm(samples,torch.transpose(centers,0,1)).clamp_(min=-1,max=1)
    ans.mul_(1. / bandwidth)
    return ans

def logistic_laplacian_model(X_train,y_train,X_val,y_val,X_test=None,y_test=None):
    y_train_sgn = np.sign(y_train)
    y_val_sgn = np.sign(y_val)

    y_train_positive_indices = [i for i in range(len(y_train_sgn)) if y_train_sgn[i] >= 0]
    y_val_positive_indices = [i for i in range(len(y_val_sgn)) if y_val_sgn[i] >= 0]

    y_train = y_train.reshape((y_train.shape[0],1))
    # y_test = y_test.reshape((y_test.shape[0],1))
    y_val = y_val.reshape((y_val.shape[0],1))

    X_train_pos,X_train_neg = X_train[y_train_positive_indices], X_train[[i for i in range(len(y_train_sgn)) if i not in y_train_positive_indices]]
    y_train_pos,y_train_neg = y_train_sgn[y_train_positive_indices], y_train_sgn[[i for i in range(len(y_train_sgn)) if i not in y_train_positive_indices]]

    X_val_pos,X_val_neg = X_val[y_val_positive_indices], X_val[[i for i in range(len(y_val_sgn)) if i not in y_val_positive_indices]]
    y_val_pos,y_val_neg = y_val_sgn[y_val_positive_indices], y_val_sgn[[i for i in range(len(y_val_sgn)) if i not in y_val_positive_indices]]

    reg = LogisticRegression().fit(X_train, y_train_sgn)
    print("training accuracy")
    training_ratio = (len(y_train_pos))/(len(y_train_pos) + len(y_train_neg))
    print(reg.score(X_train_pos, y_train_pos) * training_ratio + reg.score(X_train_neg, y_train_neg) * (1-training_ratio))
    print("test accuracy")
    val_ratio = (len(y_val_pos))/(len(y_val_pos) + len(y_val_neg))
    print(reg.score(X_val_pos, y_val_pos) * val_ratio + reg.score(X_val_neg, y_val_neg) * (1-val_ratio))

    n_class = y_train.shape[1]
    torch.backends.cudnn.enabled = True
    cuda1 = torch.cuda.current_device()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: {}".format(device))
    print("KERNEL REGRESSION")
    model = eigenpro.FKR_EigenPro(lambda x,y,samples_norm=None, centers_norm=None: kernel.laplacian(x, y,bandwidth=1/50), X_train, n_class,device=device)
    res = model.fit(x_train=X_train, y_train=y_train_sgn, x_val=X_val, y_val=y_val_sgn,epochs=list(range(30)),mem_gb=12)
