# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:18:32 2019

@author: JC
"""
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse.linalg import spsolve
from CLA_P2_Q2 import *
from rexi_coefficients import *
import matplotlib.pyplot as plt

def find_M(M,n,T=2.5,H=10):
    
    dx = H/n+1
    #zero_mat = np.zeros(n)
    I_mat = np.identity(n)
    minus_K =((1/dx)**2) *(-2 * I_mat) + np.diag(1 * np.ones(n-1),1) + np.diag(1 *np.ones(n-1),-1)
    L_mat = np.zeros((2*n,2*n))
    L_mat[n:,0:n] = minus_K
    L_mat[0:n,n:] = I_mat
    L_mat = L_mat /(dx*dx)
    eigval = np.real(np.linalg.eigvals(L_mat))
    mu = max(abs(eigval))
    h = (1.1 * T * mu)/M
    
    return h, -minus_K
"""
def band_fwd(L,b,p):
    x = np.zeros(L.shape[1])
    x[0] = b[0]/L[0,0]
    
    for k in range(1,L.shape[1]):
        j = max(1,k-p+1)
        x_k = b[k] - (npL[k,j:k]  
 """   

def band_solve(A,b):
    (L,U) = bandedLU(A,1,1)
    z = spsolve_triangular(L,b)
    #x = spsolve_triangular(U,z, lower = False)

    x = spsolve(U,z)
    #z = band_fwd(L,b)
    #x = band_bwd(U,z)
    
    return x



def wave_solve(M,n,T=2.5,H=10):
    
    #getting h for rexi coefficients
    #saving K for later
    (h,K) = find_M(M,n,T,H)
    
    #geting alpha and beta coefficients
    coef = RexiCoefficients(h,M)
    alpha = coef[0] 
    beta = coef[1]
    
    #set 0 so we can loop to sum
    UT = 0
    #generate U(0), W(0) is 0
    U0 = np.zeros(n) # w(0) are zero        
    for i in range(n):
        U0[i] = np.exp(((-np.square((i-5))))/0.2) - np.exp(-125)
    
    
    for i in range(len(alpha)):
        #A is the banded matrix from part 4
        A = (alpha[i]*alpha[i])*(np.identity(n)) + T*T*(K)
        A = sp.sparse.csr_matrix(A) #make sparse since bandedLU need sparse
        
        b = (alpha[i] * U0) #- (T* w_0) is zero from initial condition
       # b = sp.sparse.csr_matrix(b)
        #b = sp.sparse.csr_matrix(b)
        
        U_j = band_solve(A,b)
        BU_j= beta[i] * U_j
        UT = UT +  BU_j
       
    plt.plot(UT)
    plt.xlabel("distance along wave with unit H/n+1")
    plt.ylabel("U(T=2.5)")
    plt.savefig('flig20.png') #saving figures

    plt.show()
    return UT
   
wave_solve(30,100)

def rungekutta(M,n,timestep,T=2.5,H=10):
    
    dx = H/n+1
    I_mat = np.identity(n)
    minus_K =((1/dx)**2) *(-2 * I_mat) + np.diag(1 * np.ones(n-1),1) + np.diag(1 *np.ones(n-1),-1)
    L_mat = np.zeros((2*n,2*n))
    L_mat[n:,0:n] = minus_K
    L_mat[0:n,n:] = I_mat
    L_mat = L_mat /(dx*dx)
    
    U0 = np.zeros(2*n) # w(0) are zero        
    for i in range(n):
        U0[i] = np.exp(((-np.square((i-5))))/0.2) - np.exp(-125)
    U = U0
    for n in range(n):
        Uhalf = U + (timestep/2) * (L_mat @ U)
        U = U + (timestep/2) * (L_mat @ Uhalf)
    
    plt.plot(U)
    plt.savefig('rkfig30.png') #saving figures


#L_mat = np.block([[zero_mat,I_mat],[minus_K,zero_mat]])

"""
zero_mat = sparse.diags(np.zeros(n),0,format='csr')
I_mat = sparse.diags(np.ones(n),0,format='csr')

minus_K = 2 * I_mat + sparse.diags(-1 * np.ones(9),1) +  sparse.diags(-1 * np.ones(9),-1)

L_mat = sp.bmat([[zero_mat,I_mat],[minus_K,zero_mat]]) #*1j *-1j




#zero_mat = sparse.coo_matrix(np.diag(np.zeros(n),0,format='csr'))
#I_mat = sparse.coo_matrix(np.diag(np.ones(n),0,format='csr'))

minus_K = 2 * I_mat + sparse.diags(-1 * np.ones(9),1) +  sparse.diags(-1 * np.ones(9),-1)

L_mat = sp.bmat([[zero_mat,I_mat],[minus_K,zero_mat]]) #*1j *-1j
"""
