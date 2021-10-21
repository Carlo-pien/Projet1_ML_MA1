import numpy as np

def add_offset(tX):
    return np.concatenate([np.ones((tX.shape[0], 1)), tX], axis=1)

def loss(y, tX, w):
    N = tX.shape[0]
    residuals = y - tX@w
    return -np.dot(residuals, residuals)/(2*N)

def grad(y, tX, w):
    N = tX.shape[0]
    residuals = y - tX@w
    return -(np.dot(np.transpose(tX), residuals))/N
        
def step(y, tX, w, lr, batch_size, return_grads=False):
    ind = np.random.choice(np.arange(tX.shape[0]+1), batch_size)-1
    batch_X, batch_y = tX[ind, :], y[ind]
    grads = grad(batch_y, batch_X, w)
    if return_grads:
        return w-lr*grads, grads
    return w-lr*grads

def SGD(y, tX, w_init, lr, num_epochs, print_stuff, batch_size=1, return_losses=False):
    w = w_init
    losses = np.empty((num_epochs,))
    for epoch in range(num_epochs):
        if epoch%print_stuff==0:
            w, grads = step(y, tX, w, lr, batch_size, return_grads=True)
            losses[epoch] = loss(y, tX, w)
            print('epoch : ', epoch, ', weights : ', w, ', grads : ', grads, ', loss : ', loss(y, tX, w))
            #print('epoch : ', epoch, ', loss : ', loss(y, tX, w))
        else :
            w = step(y, tX, w, lr, batch_size)
            losses[epoch] = loss(y, tX, w)
    if return_losses :
        return w, losses
    return w

def decorrelate_features(x, threshold=.9):
    corr = np.corrcoef(np.transpose(x))
    row = np.expand_dims(np.where(corr >= threshold)[0],axis=1)
    col = np.expand_dims(np.where(corr >= threshold)[1],axis=1)
    indices = np.concatenate([row, col], axis=1)
    indices = indices[indices[:,0] < indices[:, 1]]

    to_delete = np.array([], dtype = np.int32)
    for i in range(indices.shape[0]):
        if not(indices[i, 1] in to_delete):
            index = int(indices[i, 1])
            to_delete = np.concatenate([to_delete, np.array([index])])
    to_delete.astype(np.int64)
    return np.delete(x, to_delete, axis=1), to_delete

def delete_zero_var_features(x):
    var = np.var(x, axis=0)
    to_keep = np.nonzero(var)
    return np.squeeze(x[:, to_keep], axis=1)

def delete_outliers(x, threshold):
    #bool_vect = np.all(x>threshold, axis=1)
    bool_vect = np.all(x>threshold, axis=1)
    legit_ids = np.where(bool_vect)
    print(len(legit_ids))
    clean_x = x[legit_ids, :]
    return clean_x, legit_ids

def polynomial_embedding(x, degree=2):
    if degree==0 or degree==1 :
        return x
    res = x
    for i in range(1, degree):
        pows = (i+1)*np.ones((x.shape[1],))
        res = np.concatenate((res, np.power(x, pows)), axis=1)
    return res

def standardize(x):
    m = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x-m)/std

def PCA(X, threshold=.9):
    """returns the data projected on the principal components vectors, and these principal components.
    
    arguments :
    X : np.array, the input data matrix
    threshold : float \in [0, 1], the desired  percentage of variance explained by the PCs
    
    returned values :
    PCX : transformed and reduced data
    Y : rotation matrix
    ids : kept components"""
    
    X = standardize(X)
    
    cov_mat = np.cov(X, rowvar=False)
    eig = np.linalg.eigh(cov_mat)
    eigvals = np.flip(eig[0])
    D = np.diag(eigvals)
    P = eig[1]
    Y = X@P

    eigsum = np.cumsum(eigvals)
    t = threshold * eigsum[-1]
    PC_ids = np.where(eigsum<t)
    Y_ = np.squeeze(Y[:, PC_ids], axis=1)
    P_ = np.squeeze(P[:, PC_ids], axis=1)
    return Y_, P, PC_ids

def divide_in_groups(x):
    
    group00 = np.where(x[:, 0]==-999)[0]
    group01 = np.where(x[:, 0]!=-999)[0]
    group1 = np.where(x[:, 22]==0)[0]
    group2 = np.where(x[:, 22]==1)[0]
    group3 = np.where(x[:, 22]>1)[0]
    
    ids_01 = np.intersect1d(group00, group1)
    ids_02 = np.intersect1d(group00, group2)
    ids_03 = np.intersect1d(group00, group3)
    ids_11 = np.intersect1d(group01, group1)
    ids_12 = np.intersect1d(group01, group2)
    ids_13 = np.intersect1d(group01, group3)
    ids = (ids_01, ids_02, ids_03, ids_11, ids_12, ids_13)
    
    x01 = x[ids_01, :]
    x02 = x[ids_02, :]
    x03 = x[ids_03, :]
    x11 = x[ids_11, :]
    x12 = x[ids_12, :]
    x13 = x[ids_13, :]
    
    delete_from_gr1 = [4, 5, 6, 12, 23, 24, 25, 26, 27, 28]
    delete_from_gr2 = [4, 5, 6, 12, 25, 26, 27, 28]
    
    x01 = np.delete(x01, delete_from_gr1, axis=1)
    x11 = np.delete(x11, delete_from_gr1, axis=1)
    x02 = np.delete(x02, delete_from_gr2, axis=1)
    x12 = np.delete(x12, delete_from_gr2, axis=1)
    x01 = np.delete(x01, 0, axis=1)
    x02 = np.delete(x02, 0, axis=1)
    x03 = np.delete(x03, 0, axis=1)
    
    data = (x01, x02, x03, x11, x12, x13)
    
    return data, ids
    
    
            
        
    
    
    
    
   