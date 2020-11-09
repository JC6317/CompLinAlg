# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:28:23 2019

@author: JC
CID:01063446
"""

from CLA_P1_Q1 import householderQR
import numpy as np

test1 = np.array([[1.,0.5,1/3],[0.5, 1/3,0.25],[1/3,0.25,0.2],[1/4,1/5,1/6]])


def QRtester(test):

    (Q,R,A) = householderQR(test)
    
    #Q times Q hermitian conjugate should equal identity
    Q_transpose = np.transpose(Q)
    Q_star = np.conjugate(Q_transpose)
    I_test1 = np.allclose(np.dot(Q,Q_star),np.identity(Q.shape[0]))
    I_test2 = np.allclose(np.dot(Q_star,Q),np.identity(Q.shape[0]))
    if I_test1 == False or I_test2 == False:
        I_test = False
    else:
        I_test = True
    
    #check entries of R
    uptritest= True #set test result to true by default
    for ro in range(1,R.shape[0],1):
        colrange = min(ro, R.shape[1]) #take the minimum since R can be rectangular
        for co in range(colrange):
            if abs(R[ro,co]) >1e-13: #checking if value is effectively 0
                uptritest = False #change test result to false if non zero value exists
                break
        else:
            continue
        break
    
    A_test = np.allclose(np.dot(Q,R),A) # check Q and R remultiply to form A
    
    testresult = False #assume testresult is false
    if I_test == True and uptritest == True and A_test == True:
        testresult = True #change result to true only if all 3 tests are true
    
    return testresult

print(QRtester(test1))

test2 = np.array([[1.,1.j,0],[2.+3.j,0,0],[1.+1.j,9,0.2j],[4., 0.2j, 0]])
print(QRtester(test2))

bulktests = np.empty(10) #array of results
for testmat in range(10):
    m = np.random.randint(1,10)
    n = np.random.randint(1,m+1) #ensure m>n
    bulktests[testmat] = QRtester(np.random.rand(m,n)) #fill in results array
bulktests
print(np.sum(bulktests) == bulktests.shape) #check every test was true
      
