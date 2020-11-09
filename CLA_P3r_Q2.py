# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 18:13:07 2019

@author: JC
"""

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse import linalg
import matplotlib.pyplot as plt
'''
q2
A is symmetric so it is diagonalisable 
'''
def A2_construct(dim,N,B):#dimension of matrix position of outer diagonals
    diagonals = [2*(1+B)*np.ones(dim),-1*np.ones(dim-1),-1*np.ones(dim-1),-B*np.ones(dim-N),-B*np.ones(dim-N)]
    A = sp.csr_matrix(sp.diags(diagonals, [0,-1,1,N,-N]))
    return A 

np.linalg.cond(A2_construct(100,100,1).toarray())

def GMRES_A(dim,N,B,t=1e-05): #t tolerance
    A = A2_construct(dim,N,B)
    b = np.random.normal(0,10,dim)
    
    counter = gmres_callback
    
    (x,info) = sp.linalg.gmres(A,b,tol=t)
    return x,info
#modify N does convergence speed up, if not, we need preconditioner
    


def N_converge(N):   
    return scipy.linalg.eig(A2_construct(100,N,1).toarray())[0]

eigplot = np.zeros((7,100))
for n in range(7):
    eigplot[n] = N_converge(5+ n*15)
plt.scatter(eigplot[0],np.zeros(100))
#eigenvalues scattered for all N
#so convergence should be slow for all N
#potentially a bit quicker if N is larger? evals more compact, but still slow
#need to check residuals

def e_vec_N(n,b=1):
    An= A2_construct(100,n,b)
    evaln = sp.linalg.eigs(A10, k=n)[0]
    plt.plot(evaln)
'''
Q3
'''    
def split_solve(dim,n,x0,iterations,gamma,b,B):
    #b = np.random.normal(0,10,dim)

    diagonals_M = [2*np.ones(dim),-1*np.ones(dim-1),-1*np.ones(dim-1)] #for M matrix
    diagonals_N = [2*B*np.ones(dim),-B*np.ones(dim-n),-B*np.ones(dim-n)] #for N matrix
    M = sp.csr_matrix(sp.diags(diagonals_M, [0,-1,1])) #generate M
    N = sp.csr_matrix(sp.diags(diagonals_N, [0,-n,n])) #generate N
    
    Ahalf = gamma*sp.identity(dim,format="csr") + M #the matrix system to be solved for the half step
    Afull = gamma*sp.identity(dim,format="csr") + N #matrix system to be solved for full step
    x_n = x0 #initialise x_n
    X=np.zeros((iterations+1,dim)) #create an empty matrix to fill with x vector solutions into rows
    X[0,:]= x0 #first row is the initial vector, perhaps make this column instead
    
    for i in range(iterations):
        bhalf = (gamma*sp.identity(dim,format="csr") - N).dot(x_n) + b #create b vector for half step solve

        x_n = sp.spsolve(Ahalf,bhalf) #this is now x_n+1/2
        
        bfull = (gamma*sp.identity(dim,format="csr") - M).dot(x_n) + b #create b vector (RHS) for full step, x_n here represnets the half step solution
        
        x_n = sp.spsolve(Afull,bfull) #full step solution
        
        X[i+1] = x_n #save this full step into X matrix
    
    return X
"""
larger b means smaller result
"""
def iterate(r_k,dim,B,n,gamma):
    z_k = split_solve(dim,n,0,1,gamma,r_k,B)
    return z_k

def iterate_long(r_k,dim,B,n,gamma):
    diagonals_M = [2*np.ones(dim),-1*np.ones(dim-1),-1*np.ones(dim-1)] #for M matrix
    diagonals_N = [2*B*np.ones(dim),-B*np.ones(dim-n),-B*np.ones(dim-n)] #for N matrix
    M = sp.csr_matrix(sp.diags(diagonals_M, [0,-1,1])) #generate M
    N = sp.csr_matrix(sp.diags(diagonals_N, [0,-n,n])) #generate N
    
    Ahalf = gamma*sp.identity(dim,format="csr") + M #the matrix system to be solved for the half step
    Afull = gamma*sp.identity(dim,format="csr") + N #matrix system to be solved for full step
    
    y = sp.spsolve(Ahalf,r_k)
    
    bfull = (gamma*sp.identity(dim,format="csr") - M).dot(y) + r_k
    z_k = sp.spsolve(Afull,bfull)
    
    return z_k

def GMRES_p(b,dim,B,n,gamma):#preconditioned GMRES
    #what is r_k
    A = A2_construct(dim,n,B)
    
    def iterate(r_k):
        z_k = split_solve(dim,n,0,1,gamma,r_k,B)
        return z_k
    
    M = sp.linalg.LinearOperator((dim**2,dim**2), matvec=iterate)
    x = sp.linalg.gmres(A,b,M=M)
    
    return x
    
    