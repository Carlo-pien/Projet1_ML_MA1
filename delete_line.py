# Useful starting lines
import numpy as np

def delete(tx,a):
    j = 0
    for i in range(tx.shape[0]):
        if a in tx[i-j,:]:
            tx = np.delete(tx,i-j,0)
            j = j+1
    return tx
            
        
    