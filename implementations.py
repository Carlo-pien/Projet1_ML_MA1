# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2

#Least squares regression thanks to normal equations
def least_squares(y, tx):
    #compute with the normal equations
    tx_t = np.transpose(tx)
    w = np.linalg.solve(tx_t.dot(tx), tx_t.dot(y))
    
    #compute the loss
    N = len(y)
    e = y-tx.dot(w)
    loss = 1/(2*N) * (np.transpose(e)).dot(e)
    return w, loss

#Least squares regression using the gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    #computation of w
    ws = [initial_w]
    w = initial_w
    for n_iter in range(max_iters):
        #computation of the gradient and of the loss
        e = y-tx.dot(ws[n_iter])
        N = len(y)
        gradient = -1/N * np.transpose(tx).dot(e)
        loss = 1/(2*N) * (np.transpose(e)).dot(e)
       
        #Update of w
        w = ws[n_iter]-gamma*gradient
        # store w
        ws.append(w)
        
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return w, loss