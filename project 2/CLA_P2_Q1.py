# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 01:07:07 2019
print(np.finfo(np.float).eps)
@author: JC
"""
import numpy as np
from matplotlib import pyplot as plt #plotting module


def random_QR(mean=0,sd=1,dim=20):
    #randomly generating B and C matrices
    B_mat = np.random.normal(mean,sd,(dim,dim))
    C_mat = np.random.normal(mean,sd,(dim,dim))
    #complex values?
    
    #Find Q from B=QR
    Q = np.linalg.qr(B_mat)[0]
    
    #Find R from C=QR
    R = np.triu(C_mat)
    
    #multipling QR to form A
    A_mat = np.matmul(Q,R)
    
    #use linalg.qr to create QR
    (Q2,R2) = np.linalg.qr(A_mat)
    
    Q2_less_Q = np.linalg.norm(Q2-Q)
    R2_less_R = np.linalg.norm(R2-R)
    Q2R2_less_A = np.linalg.norm(np.dot(Q2,R2)-A_mat)
    
    #time to complete
    #back stab
    
    return Q2_less_Q, R2_less_R, Q2R2_less_A

np.random.seed(0)
random_QR(0,1)

def random_QR_test(iterations,mean=0,sd=1,dim=20):
    results = np.zeros((iterations,3))
    for i in range(iterations): #number of iterations
        results[i,:] = random_QR(mean,sd,dim) #function input is used for this nested function
    #graph plot    
    plt.plot(np.linspace(0,iterations-1,iterations),np.log10(results[:,0]),label = 'Q residual')
    plt.plot(np.linspace(0,iterations-1,iterations),np.log10(results[:,1]),label = 'R residual')
    plt.plot(np.linspace(0,iterations-1,iterations),np.log10(results[:,2]),label = 'QR residual')
    plt.xlabel("Iteration")
    plt.ylabel("log base 10 residual")
    plt.legend(loc='upper right')
    #plt.savefig('flig2.png') #saving figures
    return results

np.random.seed(0)
random_QR_test(50)

def accuracy_check(seed):
    np.random.seed(seed)
    B_mat = np.random.normal(0,1,(20,20))
    C_mat = np.random.normal(0,1,(20,20))
    Q = np.linalg.qr(B_mat)[0]
    R = np.triu(C_mat) 
    A_mat = np.matmul(Q,R)
    (Q2,R2) = np.linalg.qr(A_mat) #all of the above is straight from random_QR
    Q3 = Q + (1e-10 * np.random.normal(0,1,(20,20))) #used 1e-10 because 1e-9 caused Q3 and R3 residual to be larger than Q2 and R2 residual
    R3 = R + (1e-10 * np.random.normal(0,1,(20,20))) #and we want to show with smaller Q3 and R3 residuals, the Q3R3 residual can be bigger
    return np.linalg.norm(Q3-Q), np.linalg.norm(R3-R), np.linalg.norm(A_mat - np.matmul(Q3,R3))

for i in range(5):
    print(accuracy_check(i))

def random_SVD(mean=0,sd=1,dim=20):
    #randomly generating B and C matrices
    B_mat = np.random.normal(mean,sd,(dim,dim))
    #C_mat = np.random.normal(mean,sd,dim)
    C_mat = np.random.normal(mean,sd,(dim,dim))
    
    D_mat = np.random.normal(mean,sd,(dim,dim))
    #complex values?
    
    #Find U from B
    U = np.linalg.svd(B_mat)[0] #index 0 aka 1st output is the U matrix
    
    #sigma
    #sigma_vec = np.array([abs(a) for a in C_mat])
    #sigma = np.diag(np.sort(sigma_vec))  
    sigma = np.diag(np.linalg.svd(C_mat)[1]) #index 1 aka second output of svd is sigma vec array which needs to be converted into diag matrix
    
    #Find V from C
    V = (np.linalg.svd(D_mat)[2]) #index 2 aka third output is the V matrix
    
    
    #multipling QR to form A
    A_mat = np.matmul(np.matmul(U,sigma),V)
    
    #use linalg.qr to create QR
    (U2,sigma2_vec,V2) = np.linalg.svd(A_mat)
    sigma2 = np.diag(sigma2_vec)
    
    U2_less_U = np.linalg.norm(U2-U) #2 norm is error
    U3= numpy.multiply(U2,np.divide(U2,U)) # element wise multiply by 1 or -1 to fix sign error
    U3_less_U = np.linalg.norm(U3-U)

    sigma2_less_sigma = np.linalg.norm(sigma2-sigma)
    
    V2_less_V = np.linalg.norm(V2-V)
    V3= numpy.multiply(V2,np.divide(V2,V)) # element wise multiply by 1 or -1 to fix sign error
    V3_less_V = np.linalg.norm(V3-V)

    U2sigma2V2_less_A = np.linalg.norm(np.dot(np.dot(U2,sigma2),V2)-A_mat)
    
    return U2_less_U, sigma2_less_sigma, V2_less_V, U2sigma2V2_less_A,U3_less_U,V3_less_V

random_SVD(0,1)
#def random_SVD_test(iterations,mean=0,sd=1,dim=20):
    
def accuracy_check_svd(seed):
    np.random.seed(seed)
    B_mat = np.random.normal(0,1,(20,20))
    C_mat = np.random.normal(0,1,(20,20))
    D_mat = np.random.normal(0,1,(20,20))
    U = np.linalg.svd(B_mat)[0]
    sigma = np.diag(np.linalg.svd(C_mat)[1])
    V = (np.linalg.svd(D_mat)[2])
    A_mat = np.matmul(np.matmul(U,sigma),V)
    
    U3 = U + (1e-14 * np.random.normal(0,1,(20,20))) #used 1e-10 because 1e-9 caused Q3 and R3 residual to be larger than Q2 and R2 residual
    V3 = V + (1e-14 * np.random.normal(0,1,(20,20))) #and we want to show with smaller Q3 and R3 residuals, the Q3R3 residual can be bigger
    sigma3 = sigma + (1e-14 * np.random.normal(0,1,(20,20)))
    return np.linalg.norm(U3-U), np.linalg.norm(V3-V), np.linalg.norm(A_mat - np.matmul(np.matmul(U3,sigma3),V3))