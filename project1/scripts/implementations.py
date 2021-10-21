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

def delete_outliers(x, threshold):
    bool_vect = np.all(x>threshold, axis=1)
    legit_ids = np.where(bool_vect)
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
    
            
        
    
    
    
    
   