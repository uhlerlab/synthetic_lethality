# '''Construct kernel model with EigenPro optimizer.'''
import collections
import time
import torch
import pandas as pd
from sklearn.metrics import r2_score

import torch.nn as nn
import numpy as np
import hickle as hkl

from EigenProPytorch import svd
from EigenProPytorch import utils

"""
The below is a modified implementation of https://github.com/EigenPro/EigenPro-pytorch, where some operations are batched for this problem.
In particular, we exploit the structure of the embeddings to train this model on 17 million samples without ever materializing the kernel matrix for all the samples.
"""

def asm_eigenpro_fn(sample_ids, kernel_mat, top_q, bs_gpu, alpha, min_q=5, seed=1):
    """Prepare gradient map for EigenPro and calculate
    scale factor for learning ratesuch that the update rule,
        p <- p - eta * g
    becomes,
        p <- p - scale * eta * (g - eigenpro_fn(g))
    Arguments:
        samples:	matrix of shape (n_sample, n_feature).
        map_fn:    	kernel k(samples, centers) where centers are specified.
        top_q:  	top-q eigensystem for constructing eigenpro iteration/kernel.
        bs_gpu:     maxinum batch size corresponding to GPU memory.
        alpha:  	exponential factor (<= 1) for eigenvalue rescaling due to approximation.
        min_q:  	minimum value of q when q (if None) is calculated automatically.
        seed:   	seed for random number generation.
    Returns:
        eigenpro_fn:	tensor function.
        scale:  		factor that rescales learning rate.
        top_eigval:  	largest eigenvalue.
        beta:   		largest k(x, x) for the EigenPro kernel.
    """

    np.random.seed(seed)  # set random seed for subsamples
    start = time.time()
    n_sample = len(sample_ids)

    if top_q is None:
        svd_q = min(n_sample - 1, 1000)
    else:
        svd_q = top_q

    eigvals, eigvecs = svd.nystrom_kernel_svd(sample_ids, kernel_mat.cpu().data.numpy(), svd_q)

    if top_q is None:
        max_bs = min(max(n_sample / 5, bs_gpu), n_sample)
        top_q = np.sum(np.power(1 / eigvals, alpha) < max_bs) - 1
        top_q = max(top_q, min_q)

    eigvals, tail_eigval = eigvals[:top_q - 1], eigvals[top_q - 1]
    eigvecs = eigvecs[:, :top_q - 1]

    device = sample_ids.device
    eigvals_t = torch.tensor(eigvals.copy()).to(device)
    eigvecs_t = torch.tensor(eigvecs).to(device)
    tail_eigval_t = torch.tensor(tail_eigval, dtype=torch.float).to(device)

    scale = utils.float_x(np.power(eigvals[0] / tail_eigval, alpha))
    diag_t = (1 - torch.pow(tail_eigval_t / eigvals_t, alpha)) / eigvals_t

    def eigenpro_fn(grad, kmat):
        '''Function to apply EigenPro preconditioner.'''
        return torch.mm(eigvecs_t * diag_t,
                        torch.t(torch.mm(torch.mm(torch.t(grad),
                                                  kmat),
                                         eigvecs_t)))

    print("SVD time: %.2f, top_q: %d, top_eigval: %.2f, new top_eigval: %.2e" %
          (time.time() - start, top_q, eigvals[0], eigvals[0] / scale))

    knorms = 1 - np.sum(eigvecs ** 2, axis=1) * n_sample
    beta = np.max(knorms)

    return eigenpro_fn, scale, eigvals[0], utils.float_x(beta)
    

class FKR_EigenPro(nn.Module):
    '''Fast Kernel Regression using EigenPro iteration.'''
    def __init__(self, cell_distances, kernel_fn,cell_centers, num_cells, num_knockouts, y_dim, device="cuda"):
        super(FKR_EigenPro, self).__init__()
        self.n_centers, self.x_dim = cell_centers.shape
        # there are num_cells x num_knockouts number of samples
        self.n_centers *= num_knockouts

        # each sample has num_cell_features + num_knockouts number of features
        self.x_dim += num_knockouts

        self.device = device
        self.pinned_list = []
        self.kernel_fn = kernel_fn

        self.cell_centers = self.tensor(cell_centers, release=True)
        self.weight = self.tensor(torch.zeros(
            self.n_centers, y_dim), release=True).cpu()
        cell_distances = cell_distances.fill_diagonal_(0) # for numerical stability
        cell_distances.clamp_(min=0)
        cell_distances[cell_distances < 1e-10] = 0
        self.cell_distances = cell_distances
        self.num_cells = num_cells
        self.num_knockouts = num_knockouts

        # Laplacian kernel for samples from the same KO
        dist_ko = cell_distances.fill_diagonal_(0)
        self.K_ko = self.tensor(torch.exp(-(dist_ko)**0.5)).float()

        # Laplacian kernel for samples from different KO
        dist_all = cell_distances + 2
        self.K_all = self.tensor(torch.exp(-(dist_all)**0.5)).float()

    def __del__(self):
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")
        torch.cuda.empty_cache()

    def tensor(self, data, dtype=None, release=False):
        tensor = torch.tensor(data, dtype=dtype,
                              requires_grad=False).to(self.device)
        if release:
            self.pinned_list.append(tensor)
        return tensor

    def multiply_by_w(self,arg_samples,weight):
        """
        Batch calculate pred = K(X,arg_samples)@ weight using O(|arg_samples| x num_knockouts) extra space.
        Output is len(arg_samples) x 1.
        """
        # samples are in column-major order
        cell_ids = arg_samples % self.num_cells
        knockout_ids = arg_samples // self.num_cells

        K_ko = self.K_ko[cell_ids,:]
        K_all = self.K_all[cell_ids,:]
        
        weight = weight.reshape((self.num_knockouts,self.num_cells)).T
    
        output = K_all.cpu() @ weight
        ko_substitution = (K_ko.cpu() * weight[:,knockout_ids].T).sum(axis=1)
        output[np.arange(len(arg_samples)),knockout_ids] = ko_substitution
        return self.tensor(output.sum(axis=1).reshape((-1,1)))
    
    def materialize_kmat(self,arg_samples1,arg_samples2):
        """
        Return K(arg_samples1,arg_samples2).
        Output shape is len(arg_samples1) x len(arg_samples2), which should be kept small.
        """
        cell_ids1 = arg_samples1 % self.num_cells
        knockout_ids1 = (arg_samples1 // self.num_cells).reshape((-1,1))

        if arg_samples1 is arg_samples2:
            cell_ids2 = cell_ids1
            knockout_ids2 = knockout_ids1.reshape((1,-1))
        else:
            cell_ids2 = arg_samples2 % self.num_cells
            knockout_ids2 = (arg_samples2 // self.num_cells).reshape((1,-1))
        
        output = self.K_all[cell_ids1,:][:,cell_ids2]
        K_ko = self.K_ko[cell_ids1,:][:,cell_ids2]

        mask = knockout_ids1 == knockout_ids2
        output += K_ko*mask - output*mask

        return torch.nan_to_num(output)
        

    def forward(self, arg_samples, weight=None):
        if weight is None:
            weight = torch.nan_to_num(self.weight)
        pred = self.multiply_by_w(arg_samples,weight)

        return pred

    def primal_gradient(self, arg_samples, labels, weight):
        pred = self.forward(arg_samples, weight)
        grad = pred - labels
        return grad

    @staticmethod
    def _compute_opt_params(bs, bs_gpu, beta, top_eigval):
        if bs is None:
            bs = min(np.int32(beta / top_eigval + 1), bs_gpu)

        if bs < beta / top_eigval + 1:
            eta = bs / beta
        else:
            eta = 0.99 * 2 * bs / (beta + (bs - 1) * top_eigval)
        return bs, utils.float_x(eta)

    def eigenpro_iterate(self, eigenpro_fn,
                         eta, sample_ids, batch_ids, y_batch):
        # update random coordiate block (for mini-batch)
        grad = self.primal_gradient(batch_ids, y_batch, self.weight)
        self.weight.index_add_(0, batch_ids.cpu(), (-eta * grad).cpu())

        kmat = self.materialize_kmat(batch_ids,sample_ids)
        correction = eigenpro_fn(grad, kmat)
        self.weight.index_add_(0, sample_ids.cpu(), (eta * correction).cpu())

        del kmat
        return

    def evaluate(self, eval_ids, y_eval, bs, x_info=None,type=None,
                 metrics=('mse', 'multiclass-acc',"r2", "sign-acc")):
        p_list = []
        n_sample, _ = y_eval.shape
        n_batch = n_sample / min(n_sample, bs)
        y_eval = y_eval.cpu().data.numpy()
        eval_ids = np.array(eval_ids)
        for batch_ids in np.array_split(range(n_sample), n_batch):
            p_batch = self.forward(eval_ids[batch_ids]).cpu().data.numpy()
            p_list.append(p_batch)
        p_eval = np.vstack(p_list)
        eval_metrics = collections.OrderedDict()
        if 'mse' in metrics:
            differences = np.square(p_eval - y_eval)
            differences_sorted = np.argsort(differences,axis=None)
            eval_metrics['mse'] = np.mean(differences)
        if 'multiclass-acc' in metrics:
            y_class = np.argmax(y_eval, axis=-1)
            p_class = np.argmax(p_eval, axis=-1)
            eval_metrics['multiclass-acc'] = np.mean(y_class == p_class)

        if 'r2' in metrics:
            eval_metrics["r2"] = r2_score(y_eval, p_eval)

        if 'sign-acc' in metrics:
            pos_index = [i for i in range(len(y_eval)) if y_eval[i] >= 0]
            neg_index = [i for i in range(len(y_eval)) if y_eval[i] < 0]

            pos_ratio = len(pos_index) / (len(y_eval))
            eval_metrics['sign-acc'] = np.mean(np.sign(p_eval[pos_index]) == y_eval[pos_index]) * (
                        pos_ratio / len(y_eval)) + np.mean(np.sign(p_eval[neg_index]) == y_eval[neg_index]) * (
                                                   1 - pos_ratio / len(y_eval))

        return eval_metrics

    def fit(self, y_train, epochs, mem_gb, x_test=None, y_test=None,sign=False,
            train_info=None, val_info=None,test_info=None,
            n_subsamples=None, top_q=None, bs=None, eta=None,
            n_train_eval=5000, run_epoch_eval=True, scale=1, seed=1):

        n_samples, n_labels = y_train.shape
        if n_subsamples is None:
            if n_samples < 100000:
                n_subsamples = min(n_samples, 1000)
            else:
                n_subsamples = 5000 # change back to 12000 after testing
            
        self.gpu_device = self.device

        metric = "sign-acc" if sign else "r2"

        mem_bytes = (mem_gb - 1) * 1024 ** 3  # preserve 1GB

        n_centers_for_memory = self.n_centers // 10
        n_subsamples_for_memory = n_subsamples

        bsizes = np.arange(n_subsamples_for_memory)
        mem_usages = ((self.x_dim + 3 * n_labels + bsizes + 1)
                      * n_centers_for_memory + n_subsamples_for_memory * 1000) * 4
        bs_gpu = np.sum(mem_usages < mem_bytes)  # device-dependent batch size

        # Calculate batch size / learning rate for improved EigenPro iteration.
        np.random.seed(seed)
        sample_ids = np.random.choice(n_samples, n_subsamples, replace=False)
        sample_ids = self.tensor(sample_ids)

        kmat = self.materialize_kmat(sample_ids,sample_ids)
        eigenpro_f, gap, top_eigval, beta = asm_eigenpro_fn(
            sample_ids,kmat, top_q, bs_gpu, alpha=.95, seed=seed)
        del kmat

        new_top_eigval = top_eigval / gap

        if eta is None:
            bs, eta = self._compute_opt_params(
                bs, bs_gpu, beta, new_top_eigval)
        else:
            bs, _ = self._compute_opt_params(bs, bs_gpu, beta, new_top_eigval)

        print("n_subsamples=%d, bs_gpu=%d, eta=%.2f, bs=%d, top_eigval=%.2e, beta=%.2f" %
              (n_subsamples, bs_gpu, eta, bs, top_eigval, beta))
            
        eta = self.tensor(scale * eta / bs, dtype=torch.float)
        
        # Subsample training data for fast estimation of training loss.
        ids = np.random.choice(n_samples,
                               min(n_samples, n_train_eval),
                               replace=False)
        y_train_eval = y_train[ids]

        res = dict()
        initial_epoch = 0
        train_sec = 0  # training time in seconds
        max_test = -float("inf")
        last_epoch = 0
        for epoch in epochs:
            start = time.time()
            last_epoch = epoch
            for _ in range(epoch - initial_epoch):
                epoch_ids = np.random.choice(
                    n_samples, int(n_samples // bs * bs), replace=False)
                for batch_ids in np.array_split(epoch_ids, n_samples / bs):
                    batch_ids = self.tensor(batch_ids)
                    y_batch = self.tensor(y_train[batch_ids])
                    self.eigenpro_iterate(eigenpro_f,
                                          eta, sample_ids, batch_ids, y_batch)
                    del y_batch, batch_ids

            if run_epoch_eval:
                train_sec += time.time() - start
                tr_score = self.evaluate(ids, y_train_eval, bs, x_info=train_info,type="train")
                tv_score = tr_score
                # x_test_norm = torch.sum(self.tensor(x_test) ** 2, dim=1, keepdim=True)
                if x_test is not None: te_score = self.evaluate(x_test, y_test, bs, x_info=test_info,type="test")
                print("train r2: {}, val r2: {}, ({} epochs, {} seconds)\t".format(
                      tr_score["r2"],
                      tv_score["r2"],
                    epoch, train_sec))

            initial_epoch = epoch
        return res