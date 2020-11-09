# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:53:12 2019

@author: JC
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

def CSR_construct(n): #n is dimension
    diagonals = [2*np.ones(n),-1*np.ones(n-1),-1*np.ones(n-1)]
    A = sp.csr_matrix(sp.diags(diagonals, [0,-1,1]))
    return A 


#A=CSR_construct(3)
#v=
def Ray_QI(n,v0,iterat):
    #v must have norm 1
    #if np.norm(v0)!= 1:
    #   break?
    #lambd = sp.csr_matrix(np.zeros(100))
    A = CSR_construct(n)
    lambd = np.zeros(iterat)
    lambd[0] = (A.dot(v0)).dot(v0)
    v = [None] * iterat
    v[0]=v0
    
    
    for k in range(1,iterat,1):
        system = A - lambd[k-1]*sp.identity(A.shape[0],format="csr")
        #exclude singular matrix
        w = sp.linalg.spsolve(system,v[k-1])
        v[k] = w/np.linalg.norm(w)
        lambd[k] = (A.dot(v[k])).dot(v[k])
        
    return v,lambd

def RQ_test(dim,n):
    A = CSR_construct(dim)
    v0 = np.random.normal(0,10,dim)
    v0 = v0/np.linalg.norm(v0)
    
    (v,lambd) = Ray_QI(A,v0,n)
    return v,lambd

def tridiag_QR(n):
    A = CSR_construct(n).toarray()
    (Q,R) = np.linalg.qr(A)
    return Q,R

def pure_QR(Adim,n):
    A = CSR_construct(Adim)
    A = A.todense()
    A_diag = np.zeros((n,Adim))
    A_diag[0] = np.diag(A)
    Q = np.eye(Adim)
    R = np.eye(Adim)
    
    for k in range(1,n,1):
        (Qk,Rk) = np.linalg.qr(A,mode="complete")
        A = Rk @ Qk
        A_diag[k] = np.diag(A)
        
        Q = Q @ Qk
        R = Rk @ R
    
    return Q,R,A_diag

def shift_QR(Adim,n,mu):
    A = CSR_construct(Adim)
    A = A.todense()
    A_diag = np.zeros((n,Adim))
    A_diag[0] = np.diag(A)
    Q = np.eye(Adim)
    R = np.eye(Adim)
    
    for k in range(1,n,1):
        (Qk,Rk) = np.linalg.qr(A-mu*np.eye(Adim),mode="complete")
        A = Rk @ Qk +mu*np.eye(Adim)
        A_diag[k] = np.diag(A)
        
        Q = Q @ Qk
        R = Rk @ R
    
    return Q,R,A_diag



