# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:25:02 2019
based on Algorithm 10.1 p73 "Numerical Linear Algebra" Tref & Bau
@author: JC
CID:01063446
"""

import numpy as np

#2-norm function
def norm(x):
    return np.sqrt(np.dot(x, np.conjugate(x)))

def householderQR(A):
    A = A.astype('complex128') #ensuring that A can take complex values
    A_copy = A.copy() # make a copy of A so that it can be compared to QR later
    (m,n)=A.shape #dimensions of A
    b = np.identity(m) #b is set to identity so that Q can be calculated
    b = b*1.j * (-1.j) #equivalent to changing datatype
    #b could probably be set as an input to the function,
    #one could then choose to set b as I, or as a set of datapoints
    for k in range (n): 
        #~~~~~~~~~~~~~~~~~~~finding R ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x = A[k:,k]
        x_2norm = norm(x)
        #like(x) gives same length as x
        e1 = np.zeros_like(x)
        e1[0]= 1
        v_k = (np.sign(x[0]) * x_2norm * e1 ) + x
        v_k_2norm = norm(v_k)
        v_k = v_k/v_k_2norm
        #k: includsive on k and remaining entries
        A[k:,k:] = A[k:,k:] - 2* np.outer(v_k,(np.dot(np.conjugate(v_k),A[k:,k:])))
        
        
        #~~~~~~~~~~~~~~~~~~~finding Q~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #0 to m-1 inclusive
        for i in range(m):
            #np.dot(np.conjugate,b[k:]) is scalar
            #calculating columns of Q, since b was set to be identity
            b[k:,i] = b[k:,i] - 2 * v_k * (np.dot(np.conjugate(v_k),b[k:,i]))
            
    b = np.transpose(b)
    b = np.conjugate(b)
    
    return b,A, A_copy

           