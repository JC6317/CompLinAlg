# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:13:34 2019

@author: JC
"""
#import modules
import numpy as np
import scipy as sp
from scipy import sparse
"""
old function, has problems
def bandedLU_old(M, m_l, m_u):
    A = M * 1.j * -1.j #quick fix to set data type
    m = A.shape[0] #check dimension of A
    U=sp.sparse.csr_matrix(A) 
    L= sp.sparse.csr_matrix(np.identity(m),dtype=np.complex128)
    
    #avoid sparse efficiency warning by filling in dummy 1s
    for i in range(m_l+1):
        L = L + sparse.diags(np.ones(m-i),-i)
    #from lecture notes
    for k in range(m-1):
        n1 = min(m,k+m_l +1)
        for j in range(k+1,min(m,n1)):
            
            L[j,k] = U[j,k]/U[k,k]
            
            n = min(k + m_u +1,m )
            
            U[j,k:n] = U[j,k:n] - L[j,k]*U[k,k:n]

    return L, U
"""
def bandedLU(M, m_l, m_u):
    U = M * 1.j * -1.j #quick fix to set data type
    m = U.shape[0] #check dimension of M
    L= sp.sparse.csr_matrix(np.identity(m),dtype=np.complex128)
    
    #tried to avoid sparse efficiency warning by filling in dummy 1s, however this caused function to fail
    #for i in range(m_l+1):
    #    L = L + sparse.diags(np.ones(m-i),-i)
        
        
    
    for k in range(m-1):
        #limit ensures unecessary zeros arent computed
        limit =min(m, k + m_l +1)

        L[k+1:limit,k] = U[k+1:limit,k]/U[k,k]
        
        for j in range(k+1,min(m,limit)):
            
            n = min(k + m_u +1,m )
            
            U[j,k:n] = U[j,k:n] - L[j,k]*U[k,k:n]

    return L, U

#test
Bmid = sp.sparse.diags(np.array([1+0.5j,1,1,2]), offsets=0, format="csr")
Blo = sp.sparse.diags(np.array([1,1,1]), offsets=-1, format="csr")
Bup = sp.sparse.diags(np.array([1,1,0.7j]), offsets=1, format="csr")
B = Bmid + Bup +Blo
(testl, testu)= bandedLU(B,1,1)
error=np.linalg.norm((B - testl.dot(testu)).toarray())
np.allclose(error, 0)
np.allclose(np.triu(testu.toarray(),0),testu.toarray())
np.allclose(np.tril(testl.toarray(),0),testl.toarray())




def bandedLUtest(dim,mean,sd):
    dim = 5 # has to be odd
    lower_band = np.random.randint(0, dim+1) #numpy randomis not inclusive on upper bound
    upper_band = np.random.randint(0, dim+1) #randomly generate a bandwidth
    
    """
    A = 0 #check csr
    for diag in range(-lowerband,upper_band,1):
        A_diag = sp.sparse.diags(np.random.normal(mean,sd,dim-diag), offsets=diag, format="csr")
        A_diag_complex = sp.sparse.diags(1j * np.random.normal(mean,sd,dim-diag), offsets=diag, format="csr")
        A = A + A_diag + A_diag_complex
    """
    A_real = np.random.normal(mean,sd,dim) #generate real matrix
    A_comp = 1j * np.random.normal(mean,sd,dim) #generate imaginary matrix
    A_np = A_real + A_comp #complex matrix
    
    A_np = np.triu(A_np,-1*lower_band) #upper and lower tri to turn into banded
    A_np= np.tril(A_np,upper_band)
    A = sp.sparse.csr_matrix(A_np) #set as a csr matrix
    B=A.copy() #save  a copy so it can be compared
    
    (L,U) = bandedLU(A,lower_band,upper_band)
    error=np.linalg.norm((B - testl.dot(testu)).toarray()) #check multiplication operator
    
    uptri_test = np.allclose(np.triu(U.toarray(),0),U.toarray())
    lowtri_test= np.allclose(np.tril(L.toarray(),0),L.toarray())
    
    return error, uptri_test, lowtri_test


    
    
