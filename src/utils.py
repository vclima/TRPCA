import numpy as np 
import scipy.fft as fft
import scipy.ifft as ifft 

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
