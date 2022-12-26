import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import inv as inv

import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import inv as inv


def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)

def cov_mat(mat):
    dim1, dim2 = mat.shape
    new_mat = np.zeros((dim2, dim2))
    mat_bar = np.mean(mat, axis = 0)
    for i in range(dim1):
        new_mat += np.einsum('i, j -> ij', mat[i, :] - mat_bar, mat[i, :] - mat_bar)
    return new_mat

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)


def mnrnd(M, U, V):
    """
    Generate matrix normal distributed random matrix.
    M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
    """
    dim1, dim2 = M.shape
    X0 = np.random.rand(dim1, dim2)
    P = np.linalg.cholesky(U)
    Q = np.linalg.cholesky(V)
    return M + np.matmul(np.matmul(P, X0), Q.T)


'''
alpha in BTMFS represent the scaling factor of additional_info
'''

def BTMFS(dense_mat, sparse_mat, additional_info, alpha ,init, rank, time_lags, multi_steps, maxiter1, maxiter2):
    """
    Bayesian Temporal Matrix Factorization with Side information
    BTMFS generates genetic factor W, temporal factor X and noise factor via Gibbs sampling.
    
    Parameters
    ----------
    dense_mat: N*T matrix with all entries are known.
    sparse_mat: N*T matrix with partial entries are known.
    additional_info: Side information matrix includes domain knowledge.
    alpha: The scaling factor in parameter estimation of genetic factors. it is used to preserve positive definiteness.
    init: Initial values of genetic factor W and temporal factor X. They are randomly generated. 
    rank: The number of temporal factors. You can set this value yourself.
    time_lags: The time lag set for vector autoregression.  
    multi_steps: The parameter used in prediction. 
    maxiter1 and maxiter2 represent the number of iterations for Gibbs sampling
    ----------
    
    Returns
    ----------
    This function returns:
    For imputation tasks:
    The matrix with imputation values, genetic factor W, temporal factor X, and coefficient (in vector autoregression) A 
    For prediction tasks:
    The matrix with imputation values and partial prediction values (Depend on the step size), genetic factor W, temporal factor X, and coefficient (in VAR) A 
    
    """
    W = init["W"]
    X = init["X"]
    G = additional_info*alpha
    
    d = time_lags.shape[0]
    dim1, dim2 = sparse_mat.shape #N*T
    dim1_w = W.shape[0]
    pos = np.where((dense_mat != 0) & (sparse_mat == 0)) 
    position = np.where(sparse_mat != 0)
    binary_mat = np.zeros((dim1, dim2))
    binary_mat[position] = 1
    
    beta0 = 1
    nu0 = rank
    mu0 = np.zeros((rank))
    W0 = np.eye(rank)
    tau = 1
    alpha = 1e-6
    beta = 1e-6
    S0 = np.eye(rank)
    Psi0 = np.eye(rank * d)
    M0 = np.zeros((rank * d, rank))
    rank_diag = np.identity(rank)
    feat_dim = G.shape[0]
    I_feature = np.eye(feat_dim)

    Psi_b0 = np.eye(feat_dim)
    M_b0 = np.zeros((feat_dim, rank))
    S_b0 = np.eye(rank)
    nu_b0 = rank
    
    W_plus = np.zeros((dim1, rank))
    X_plus = np.zeros((dim2, rank))
    A_plus = np.zeros((rank, rank, d))
    for iters in range(maxiter1):
        
        
        Psi_b = inv(inv(Psi_b0) + np.matmul(G, G.T))
        M_b = np.matmul(Psi_b, np.matmul(inv(Psi_b0), M_b0)+np.matmul(G, W))
        S_b = S_b0 + np.matmul(W.T, W) + np.matmul(np.matmul(M_b0.T, inv(Psi_b0)), M_b0) - np.matmul(np.matmul(M_b.T, inv(Psi_b)), M_b)
        Sigma_b = invwishart(df = nu_b0 + dim1, scale = S_b, seed = None).rvs()
        inv_Sigma_b = inv(Sigma_b)
        B = mnrnd(M_b, Psi_b, Sigma_b)
           
        
        var1 = X.T
        var2 = kr_prod(var1, var1)        
        var3 = tau * np.matmul(var2, binary_mat.T).reshape([rank, rank, dim1]) + np.dstack([inv_Sigma_b] * dim1)
        var4 = tau * np.matmul(var1, sparse_mat.T) + np.matmul(inv_Sigma_b, np.matmul(B.T, G))
        
        for i in range(dim1):
            inv_var_Lambda = inv(var3[:, :, i])
            W[i, :] = mvnrnd(np.matmul(inv_var_Lambda, var4[:, i]), inv_var_Lambda)
        if iters + 1 > maxiter1 - maxiter2:
            W_plus += W
        
        Z_mat0 = X[0 : np.max(time_lags), :]
        Z_mat = X[np.max(time_lags) : dim2, :]
        Q_mat = np.zeros((dim2 - np.max(time_lags), rank * d))
        for t in range(np.max(time_lags), dim2):
            Q_mat[t - np.max(time_lags), :] = X[t - time_lags, :].reshape([rank * d])
        var_Psi = inv(inv(Psi0) + np.matmul(Q_mat.T, Q_mat))
        var_M = np.matmul(var_Psi, np.matmul(inv(Psi0), M0) + np.matmul(Q_mat.T, Z_mat))
        var_S = (S0 + np.matmul(Z_mat.T, Z_mat) + np.matmul(np.matmul(M0.T, inv(Psi0)), M0) 
                 - np.matmul(np.matmul(var_M.T, inv(var_Psi)), var_M))
        Sigma = invwishart(df = nu0 + dim2 - np.max(time_lags), scale = var_S, seed = None).rvs()
        Lambda_x = inv(Sigma)
        A = mat2ten(mnrnd(var_M, var_Psi, Sigma).T, np.array([rank, rank, d]), 0)
        if iters + 1 > maxiter1 - maxiter2:
            A_plus += A

        var1 = W.T
        var2 = kr_prod(var1, var1)
        var3 = tau * np.matmul(var2, binary_mat).reshape([rank, rank, dim2]) + np.dstack([Lambda_x] * dim2)
        var4 = tau * np.matmul(var1, sparse_mat)
        for t in range(dim2):
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            if t < np.max(time_lags):
                Qt = np.zeros(rank)
            else:
                Qt = np.matmul(Lambda_x, np.matmul(ten2mat(A, 0), X[t - time_lags, :].reshape([rank * d])))
            if t < dim2 - np.min(time_lags):
                if t >= np.max(time_lags) and t < dim2 - np.max(time_lags):
                    index = list(range(0, d))
                else:
                    index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim2)))[0]
                for k in index:
                    Ak = A[:, :, k]
                    Mt += np.matmul(np.matmul(Ak.T, Lambda_x), Ak)
                    A0 = A.copy()
                    A0[:, :, k] = 0
                    var5 = (X[t + time_lags[k], :] 
                            - np.matmul(ten2mat(A0, 0), X[t + time_lags[k] - time_lags, :].reshape([rank * d])))
                    Nt += np.matmul(np.matmul(Ak.T, Lambda_x), var5)
            var_mu = var4[:, t] + Nt + Qt
            if t < np.max(time_lags):
                inv_var_Lambda = inv(var3[:, :, t] + Mt - Lambda_x + np.eye(rank))
            else:
                inv_var_Lambda = inv(var3[:, :, t] + Mt)
            X[t, :] = mvnrnd(np.matmul(inv_var_Lambda, var_mu), inv_var_Lambda)
        mat_hat = np.matmul(W, X.T)
        if iters + 1 > maxiter1 - maxiter2:
            X_plus += X
        
        tau = np.random.gamma(alpha + 0.5 * sparse_mat[position].shape[0], 
                              1/(beta + 0.5 * np.sum((sparse_mat - mat_hat)[position] ** 2)))
        rmse = np.sqrt(np.sum((dense_mat[pos] - mat_hat[pos]) ** 2)/dense_mat[pos].shape[0])
        if (iters + 1) % 200 == 0 and iters < maxiter1 - maxiter2:
            print('Iter: {}'.format(iters + 1))
            print('RMSE: {:.6}'.format(rmse))
            print()

    W = W_plus/maxiter2
    A = A_plus/maxiter2
    X_new = np.zeros((dim2 + multi_steps, rank))
    X_new[0 : dim2, :] = X_plus/maxiter2
    for t0 in range(multi_steps):
        X_new[dim2 + t0, :] = np.matmul(ten2mat(A, 0), X_new[dim2 + t0 - time_lags, :].reshape([rank * d]))
    if maxiter1 >= 100:
        final_rmse = np.sqrt(np.sum((dense_mat[pos] - mat_hat[pos]) ** 2)/dense_mat[pos].shape[0])

        return np.matmul(W, X_new[dim2 : dim2 + multi_steps, :].T), W, X_new, A, final_rmse

    
    return np.matmul(W, X_new[dim2 : dim2 + multi_steps, :].T), W, X_new, A

def multi_prediction_side(dense_mat, sparse_mat, additional_info, alpha, pred_time_steps, multi_steps, rank, time_lags, maxiter):
    """
    multi_prediction_side
    multi_prediction_side function is used for imputations and predictions.
    
    Parameters
    ----------
    dense_mat: N*T matrix with all entries are known.
    sparse_mat: N*T matrix with partial entries are known.
    additional_info: Side information matrix includes domain knowledge.
    alpha: The scaling factor in parameter estimation of genetic factors. it is used to preserve positive definiteness.
    pred_time_steps: Length of time points need to be predicted. 
    rank: The number of temporal factors. You can set this value yourself.
    multi_steps: The parameter used in prediction. 
    time_lags: The time lag set for vector autoregression.  
    maxiter: maxiter represent the number of iterations for Gibbs sampling in imputations and predictions.
    ----------
    
    Returns
    ----------
    This function returns:
    The matrix with prediction values (Thus the matrix is N by pred_time_steps), the RMSE of imputation task, the RMSE of prediction task. 
    
    """
    T = dense_mat.shape[1]
    start_time = T - pred_time_steps
    dim1 = dense_mat.shape[0]
    d = time_lags.shape[0]
    mat_hat = np.zeros((dim1, pred_time_steps))
    
    for t in range(int(pred_time_steps/multi_steps)):
        if t == 0:
            init = {"W": 0.1 * np.random.rand(dim1, rank), "X": 0.1 * np.random.rand(start_time, rank)}
            mat, W, X, A, imput_rmse = BTMFS(dense_mat[:, 0 : start_time], 
                                sparse_mat[:, 0 : start_time], 
                                additional_info, alpha, init, rank, time_lags, multi_steps, maxiter[0], maxiter[1])
        else:
            init = {"W": W, "X": X}
            mat, W, X, A = BTMFS(dense_mat[:, 0 : start_time + t * multi_steps], 
                                sparse_mat[:, 0 : start_time + t * multi_steps], 
                                additional_info, alpha, init, rank, time_lags, multi_steps, maxiter[2], maxiter[3])
        mat_hat[:, t * multi_steps : (t + 1) * multi_steps] = mat[:, mat.shape[1] - multi_steps : mat.shape[1]]

    small_dense_mat = dense_mat[:, start_time : dense_mat.shape[1]]
    pos = np.where(small_dense_mat != 0)

    final_rmse = np.sqrt(np.sum((small_dense_mat[pos] - 
                                 mat_hat[pos]) ** 2)/small_dense_mat[pos].shape[0])
    
    return mat_hat, imput_rmse, final_rmse