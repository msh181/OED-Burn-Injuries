import numpy as np

def makeColSol(x,t,T):
    x=x[1]
    t=t[:,1]
    colSol=[0.0,0.0,0.0]
    for i in range(len(t)):
        for j in range(len(x)):
            colSol = np.vstack((colSol,[x[j],t[i],T[i,j]]))
    #colSol = np.delete(colSol, (0), axis=0)
    return np.delete(colSol, (0), axis=0)