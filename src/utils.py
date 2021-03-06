import numpy as np 
from scipy.fft import fft,ifft
from scipy.linalg import svd


def transpose(X):
    """The transpose of a tensor

    Args:
        X (np.array): Tensor

    Returns:
        np.array: X.T
    """
    n1,n2,n3 = X.shape
    Xt = np.zeros(n2,n1,n3)
    Xt[:,:,0] = np.copy(X[:,:,0].T)
    if n3 > 1:
        for i in range(1,n3):
            Xt[:,:,i] = X[:,:,n3-i+1]
    return Xt

def tprod(A,B):
    """Product of two tensors

    Args:
        A (np.array): Tensor
        B (np.array): Tensor

    Returns:
        np.array: C=A*B
    """
    n1, _, n3 = A.shape
    l = B.shape[2]
    A = fft(A)
    B = fft(B)
    C = np.zeros(n1,l,n3)
    for i in range(n3):
        C[:,:,i] = A[:,:,i]*B[:,:,i]
    C = ifft(C)
    return C

def prox_l1(b,lambda_):
    """ The proximal of l1 norm: 
        min_x lambda*||x||_1+0.5*||x-b||_F^2

    Args:
        b (np.array):
        lambda_ (float): regularization term

    Returns:
        np.array: proximal of l1 norm
    """
    x = np.maximum(np.zeros(b.shape),b-lambda_)+np.min(np.zeros(b.shape),b+lambda_)

    return x

def solve_Lp_w(y,lambda_,p):
    """TO DO

    Args:
        y (np.array): [description]
        lambda_ (float): 
        p (float): power of weighted tensor Schatten p-norm
    """
    J = 4

    tau = (2*lambda_*(1-p))^(1/(2-p)) + p*lambda_*(2*(1-p)*lambda_)^((p-1)/(2-p))
    x = np.zeros(y.shape)

    # Number of zero elements after threshold
    i0 = np.argwhere(np.abs(y)>tau)

    if len(i0)>0:
        y0 = y[i0]
        t = np.abs(y0)
        lambda0 = lambda_[i0]

        for j in range(J):
            t = np.abs(t) - np.power(np.multiply(p*lambda0,t),p-1)
        x[i0] = np.multiply(np.sign(y0),t)

    return x

def prox_l1(b, lambda_):
    x = np.maximum(0,b+lambda_) + np.minimum(0,b-lambda_)
    
    return x

def etrpca_tnn_lp(X, lambda_, weight, p, tol=1e-8, max_iter=500, rho=1.1, mu=1e-4, max_mu=1e10):
    
    dim=X.shape
    L = np.zeros(dim)
    S = L.copy()
    Y = L.copy()

    for i in range(max_iter):
        Lk = L.copy()
        Sk = S.copy()

        L, tnnL = prox_tnn(-S+X-Y/mu, p)

        S = prox_l1(-L+X-Y, lambda_/mu)
        
        dY = L+S-X

        chgL = np.max(np.abs(Lk-L))
        chgS = np.max(np.abs(Sk-S))
        chg = np.max([chgL,chgS,np.max(np.abs(dY))])
        
        if chg>tol:
            break
        Y = Y + mu*dY
        mu = np.minimum(rho*mu, max_mu)

    obj = tnnL+lambda_*np.linalg.norm(S, ord=1)
    err = np.linalg.norm(dY)

    return L, S, obj, err


def prox_tnn(Y,rho,p):
    '''
    %this function is used to update E of our model,E is the tensor

    % The proximal operator of the tensor nuclear norm of a 3 way tensor
    %
    % min_X rho*||X||_*+0.5*||X-Y||_F^2
    %
    % Y     -    n1*n2*n3 tensor
    %
    % X     -    n1*n2*n3 tensor
    % tnn   -    tensor nuclear norm of X
    % trank -    tensor tubal rank of X
    '''
    n1,n2,n3 = Y.shape
    n12 = min(n1,n2)
    Y = fft(Y)
    U = np.zeros([n1,n12,n3])
    V = np.zeros([n2,n12,n3])
    S = np.zeros([n12,n12,n3])
    trank = 0
    for i in range(n3):
        U[:,:,i],s,V[:,:,i] = svd(Y[:,:,i])
        s = np.diag(s)
        s = solve_Lp_w(s, rho, p); 
        S[:,:,i] = np.diag(s)
        tranki = len(np.where(s!=0))
        trank = max(tranki,trank)
    U = U[:,1:trank,:]
    V = V[:,1:trank,:]
    S = S[1:trank,1:trank,:]

    #U = ifft(U,[],3)
    U = ifft(U)
    S = ifft(S)
    V = ifft(V)

    X = tprod(tprod(U,S),transpose(V))

    S = S[:,:,1]
    tnn = np.sum(S[:]) # return the tensor nuclear norm of X
    return X,tnn,trank
    


