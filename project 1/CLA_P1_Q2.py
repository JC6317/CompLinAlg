# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:37:11 2019

@author: JC
CID:01063446
"""
import numpy as np
from CLA_P1_Q1 import householderQR #QR factorisation function
from CLA_P1_Q1 import norm #2 norm function
from matplotlib import pyplot as plt #plotting module

#importing readings file and labelling columns
#this is used in the LSE function
readings = np.genfromtxt('readings.csv', delimiter =',')
light_data = readings[:,0] #first column of readings is the light data
temp_data = readings[:,1]

#a function that solves a system of equations
def systemsolve2(R_hat,Q_hat_star_b):
    #preallocate vector for x 
    (m,n) = R_hat.shape
    x = np.zeros(n)
    x = x *1j * (-1j) #avoid casting errors
    x[n-1] = Q_hat_star_b[n-1]/R_hat[n-1,n-1] #the bottom value of the x vector can be calculated easily since R is upper triangular
    
    for k in range(1,n,1):
        backsubsum = np.dot(R_hat[n-1-k,n-k:n],x[n-k:n]) #this is the sum of the product of the previously calculated x_i and respective R values
        x[n-1-k] = (Q_hat_star_b[n-1-k] - backsubsum)/R_hat[n-1-k,n-1-k]
    
    #this is an array that solves R_hat * x = Q_hat_star_b
    #in least squares estimation this is the coefficients of the polynomial
    return x
    
#a function that finds the interpolating polynomial coefficients    
def LSE(degree):
    #set n to deree of poly
    datapoints= np.size(light_data)
    #vandermonde matrix has a row for each datapoint and a column for each exponent
    A2_shape = (datapoints,degree + 1)
    A2 = np.zeros(A2_shape) #preallocate some space to fill in vandemonde values
    for rowcount in range(datapoints):
        for columncount in range(degree + 1):
            A2[rowcount,columncount]= np.power(light_data[rowcount],columncount)
            #column index dictates the exponent, and row index dictates the data point
    (Q_extra, R_extra, A_copy) = householderQR(A2)        
    Q_reduced = Q_extra[:,0:degree+1]
    R_reduced = R_extra[0:degree+1,0:degree+1]
    Q_reduced_adjoint = np.conjugate(Q_reduced)
    Q_reduced_adjoint = np.transpose(Q_reduced_adjoint)
    Q_hat_star_b = np.dot(Q_reduced_adjoint, temp_data)
    
    x = systemsolve2(R_reduced,Q_hat_star_b)
    
    #need to flip x since the polynomial function counts from highest degree
    x = np.flip(x)
    #convert coefficients into polynomial function
    poly = np.poly1d(x)
    #xaxis = np.linspace(np.amin(light_data),np.amax(light_data),1000)
    #plt.plot(light_data, temp_data,'x')
    #plt.plot(xaxis,poly(xaxis))
    r = poly(light_data) - temp_data
    r_2norm = norm(r)
    
    return x, r, r_2norm, poly
#x are the coefficients, r is the residue vector, r_2norm is the 2 norm of the residue, poly is the polynomial function of the x coefficients

#if __name__ == __'main'__:
for n in range(0,6,1):
    poly = LSE(n)[3] #4th output of LSE is the polynomial function
    xaxis = np.linspace(np.amin(light_data),np.amax(light_data),1000) # generate 1000 points between the minimum and maximum values of light intensity
    
    plt.plot(light_data, temp_data,'x',label='Actual data points') #marker plot of actual data points
    plt.plot(xaxis,poly(xaxis),label='interpolated polynomial of degree '+str(n)) #line plot of interpolating polynomial
    plt.xlabel("light intensity")
    plt.ylabel("temperature change")
    plt.legend(loc='upper right')
    plt.savefig('fig%1.0f.png'%(n)) #saving figures
    #alternative plt.savefig('flig'+str(n)+'.png')
    plt.show()

for n in range(9,100,10):
    poly = LSE(n)[3] #4th output of LSE is the polynomial function
    xaxis = np.linspace(np.amin(light_data),np.amax(light_data),1000) # generate 1000 points between the minimum and maximum values of light intensity
    
    plt.plot(light_data, temp_data,'x',label='Actual data points') #marker plot of actual data points
    plt.plot(xaxis,poly(xaxis),label='interpolated polynomial of degree '+str(n)) #line plot of interpolating polynomial
    plt.xlabel("light intensity")
    plt.ylabel("temperature change")
    plt.legend(loc='upper right')
    plt.savefig('fig%1.0f.png'%(n)) #saving figures
    #alternative plt.savefig('flig'+str(n)+'.png')
    plt.show()
       
residues = np.zeros(9) #empty vector to fill with 9 residue values
for deg in range(0,9,1):
    residues[deg] = LSE(deg)[2] #3rd output of LSE is the 2 norm of the residue vector
plt.plot(np.linspace(0,8,9), residues, label='2 norm of residue') #plot residues against degree
plt.xlabel("degree of interpolation")
plt.ylabel("2-norm of residue")
#plt.savefig('residuefig9.png')
plt.show()   
    
residues = np.zeros(99) #empty vector to fill with 99 residue values
for deg in range(0,99,1):
    residues[deg] = LSE(deg)[2] #3rd output of LSE is the 2 norm of the residue vector
plt.plot(np.linspace(0,98,99), residues, label='2 norm of residue') #plot residues against degree
plt.xlabel("degree of interpolation")
plt.ylabel("2-norm of residue")
#plt.savefig('residuefig99.png')
plt.show()

residues[0:9]

for index in range(residues.shape[0]):
    if residues[index] == np.amin(residues):
        print(index)

"""
residues = []
for deg in range(1,100,1):
    residues.append(LSE(deg)[2])
plt.plot(np.linspace(1,99,99),residues)

residues
plt.plot(np.linspace(1,99,99),residues)


"""  
    
    