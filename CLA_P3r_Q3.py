# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 01:47:41 2020

@author: JC
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import matplotlib.pyplot as plt
"""
def p_ij(xy):     #xy is a coordinate in numpy array form

    val = np.linalg.norm(xy - np.array([0.5,0.5])) <0.25
    return val
"""
"""
q1
"""
def generate(N):
    """
    generate an NxN image with noise
    we model noise with N90,0.15^2)
    image is constructed with the condition that the (x,y) coordinate satisfied equation (11) from the assignment
    Should be a circle centred at 
    """
    image = np.zeros((N,N))
    for x in range(N): #x coordinate is column index
        for y in range(N): #y coordinate is row index
            if np.linalg.norm([x/N,y/N] - np.array([0.5,0.5])) <0.25:
                image[y,x] = 1
    noise = np.random.normal(0,0.15,(N,N)) 
    image = image + noise
    plt.figure()
    plt.pcolor(image,cmap='gray')
    plt.title('noisy image')
    plt.axis('square')
    #to add later: change axis so that they are from 0 to 1
    #scale = N
    #ticks = tk.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    #plt.xaxis.set_major_formatter(ticks)
    #plt.yaxis.set_major_formatter(ticks)
    return image
"""
q2
"""
def A3_construct(N,B,mu):
    """
    Construct a matrix A as designed in part 3
    N is the shift of the outer diagonal and N^2 is the dimension
    B is the scaling coefficient for some of the entries
    mu is the difference between this and A2_construct
    """
    dim = N**2 #set dimension
    diagonals = [(2*(1+B)+(mu/(N+1)**2))*np.ones(dim),-1*np.ones(dim-1),-1*np.ones(dim-1),-B*np.ones(dim-N),-B*np.ones(dim-N)] #create array of diagonals
    A = sp.csr_matrix(sp.diags(diagonals, [0,-1,1,N,-N])) #form sparse matrix using diagonals
    return A 

def split_solve2(b,n,x0,iterations,gamma,B,mu):
    """
    similar to part 2 but accounts for mu delta x squared
    Function that solves Ax=b by splitting A3 into two matrices and then iterating through those
    A is generated with n is the same N as previous, B is the same beta
    b is the column vector Ax is solved against, unlike the GMRES function, the b must be specified so that we can use this function in 2.7
    I have changed the notation from N to n for the shift/dimension since N is used as a matrix
    x0 is the starting vector for the iteration
    iterations is the number of iterations
    gamma scales the identity i nthe iterative step
    mu is the difference between this and split_solve
    output X is an array where each row represents a new iteration/solution
    """
    dim = n**2
    diagonals_M = [(2+(mu/(n+1)**2))*np.ones(dim),-1*np.ones(dim-1),-1*np.ones(dim-1)] #for M matrix
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

        x_n = sp.linalg.spsolve(Ahalf,bhalf) #this is now x_n+1/2
        
        bfull = (gamma*sp.identity(dim,format="csr") - M).dot(x_n) + b #create b vector (RHS) for full step, x_n here represnets the half step solution
        x_n = sp.linalg.spsolve(Afull,bfull) #full step solution
        
        X[i+1] = x_n #save this full step into X matrix
    
    return X

def GMRES_3(b,B,n,gamma,mu):
    """
    preconditioned GMRES solves Ax=b
    A defined with n and B
    gamma to scale the identity in the preconditioning step
    mu has been added
    """
    A = A3_construct(n,B,mu) #generate A matrix
    dim= n**2 #set the dimension
    
    def iterate(r_k): #function takes in r_k and computes z_k for preconditioning
        b=r_k
        x0=np.zeros(dim)
        #x0=0
        z_k = split_solve2(b,n,x0,1,gamma,B,mu)[-1,:] #the last row of the split_solve function is the final iterated solution
        #z_k = split_solve(r_k,n,0,1,gamma,B)[-1,:] #the last row of the split_solve function is the final iterated solution
        return z_k
    
    M = sp.linalg.LinearOperator((dim,dim), matvec=iterate) 

    #x,info = sp.linalg.gmres(A,b,M=M,callback=counter) #callback for later plots
    x,info = sp.linalg.gmres(A,b,M=M)
    
    return x

N=100 #parametrise
beta=1
g=1
m=1000
def denoise(N,beta,gamma,mu):
    """
    This function will denoise the image constructed from 'generate'
    A noisy image is generated with dimension NxN and then plotted
    We then solve a system of equations using a preconditioned GMRES, gamma parameter for preconditioning
    This solution is the minimum of function (12)
    mu penalises closeness/accuracy
    beta penalises horizontal smoothness
    """
    phat = generate(N) #save the noisy image
    phatvec = (mu/(N+1)**2) * np.reshape(phat,N*N,order='F') #order F reads first index, ie columns first, so we stretch the array into a vector by pasting columns together
    pvec = GMRES_3(phatvec,beta,N,gamma,mu) #solve equations
    p_image = np.reshape(pvec,(N,N),order='F') #order F must be used again so matrix is reforemd the same way it was stretched out
    plt.figure() #start new plot so we can see both
    plt.pcolor(p_image,cmap='gray')
    plt.title('denoised image (B = %i, $\mu$=%i)' %(beta,mu))
    plt.axis('square')


denoise(100,1,1,1)
denoise(100,1,1,1000)

denoise(100,100,1,1)

denoise(100,1,1,50000)

denoise(100,10000,1,1)




"""
q3
"""

def A4_construct(N,lamb,mu):
    """
    Construct a matrix A as designed in part 3.3
    N is the shift of the outer diagonal and N^2 is the dimension
    lamb is the scaling coefficient for most of the entries
    mu is applied to the main diagonal
    """
    dim = N**2 #set dimension
    diagonals = [(4*(lamb)+(mu/(N+1)**2))*np.ones(dim),-lamb*np.ones(dim-1),-lamb*np.ones(dim-1),-lamb*np.ones(dim-N),-lamb*np.ones(dim-N)] #create array of diagonals
    A = sp.csr_matrix(sp.diags(diagonals, [0,-1,1,N,-N])) #form sparse matrix using diagonals
    return A 

def split_solve3(b,n,x0,iterations,lamb,mu):
    """
    similar to part 2 but accounts for mu delta x squared
    Function that solves Ax=b by splitting A3 into two matrices and then iterating through those
    A is generated with n is the same N as previous, B is now 1
    b is the column vector Ax is solved against, unlike the GMRES function, the b must be specified so that we can use this function in 2.7
    I have changed the notation from N to n for the shift/dimension since N is used as a matrix
    x0 is the starting vector for the iteration
    iterations is the number of iterations
    lamb scales entries
    mu is applied to main diagonal
    output X is an array where each row represents a new iteration/solution
    """
    dim = n**2
    diagonals_M = [((mu/(n+1)**2))*np.ones(dim),-lamb*np.ones(dim-1),-lamb*np.ones(dim-1)] #for M matrix
    diagonals_N = [4*np.ones(dim),-lamb*np.ones(dim-n),-lamb*np.ones(dim-n)] #for N matrix
    M = sp.csr_matrix(sp.diags(diagonals_M, [0,-1,1])) #generate M
    N = sp.csr_matrix(sp.diags(diagonals_N, [0,-n,n])) #generate N
    
    Ahalf = gamma*sp.identity(dim,format="csr") + M #the matrix system to be solved for the half step
    Afull = gamma*sp.identity(dim,format="csr") + N #matrix system to be solved for full step
    x_n = x0 #initialise x_n
    X=np.zeros((iterations+1,dim)) #create an empty matrix to fill with x vector solutions into rows
    X[0,:]= x0 #first row is the initial vector, perhaps make this column instead
    
    for i in range(iterations):
        bhalf = (gamma*sp.identity(dim,format="csr") - N).dot(x_n) + b #create b vector for half step solve

        x_n = sp.linalg.spsolve(Ahalf,bhalf) #this is now x_n+1/2
        
        bfull = (gamma*sp.identity(dim,format="csr") - M).dot(x_n) + b #create b vector (RHS) for full step, x_n here represnets the half step solution
        x_n = sp.linalg.spsolve(Afull,bfull) #full step solution
        
        X[i+1] = x_n #save this full step into X matrix
    
    return X

def GMRES_4(b,n,lamb,mu):
    """
    preconditioned GMRES solves Ax=b
    A defined with n and B
    gamma to scale the identity in the preconditioning step
    mu has been added
    """
    A = A4_construct(n,lamb,mu) #generate A matrix
    dim= n**2 #set the dimension
    
    def iterate(r_k): #function takes in r_k and computes z_k for preconditioning
        b=r_k
        x0=np.zeros(dim)
        #x0=0
        z_k = split_solve3(b,n,x0,1,lamb,mu)[-1,:] #the last row of the split_solve function is the final iterated solution
        #z_k = split_solve(r_k,n,0,1,gamma,B)[-1,:] #the last row of the split_solve function is the final iterated solution
        return z_k
    
    M = sp.linalg.LinearOperator((dim,dim), matvec=iterate) 

    #x,info = sp.linalg.gmres(A,b,M=M,callback=counter) #callback for later plots
    x,info = sp.linalg.gmres(A,b,M=M)
    
    return x


def shrink(x,g):
    """
    max of |x|-g or 0, multiplied by positive |x|
    used in equation (18), (19)
    """
    sign = x/np.abs(x) 
    mx = np.max([np.abs(x)-g,0])
    return sign*mx
#add lambda later
lamb = 1
mu = 1
N=100
iterat= 5
#first p mat
phat = generate(N) #save the noisy image
phatvec = (mu/(N+1)**2) * np.reshape(phat,N*N,order='F') #order F reads first index, ie columns first, so we stretch the array into a vector by pasting columns together
pvec = GMRES_4(phatvec,N,lamb,mu)
p = np.reshape(pvec,(N,N),order='F')

#first ds
dh = np.zeros((N+1,N)) #note that these have different dimensions
dv = np.zeros((N,N+1))

for i in range(N+1): #i is our long axis
    for j in range(N): #j is our short axis
        dh[i,j] = shrink((p[i+1,j]-p[i,j])/(N+1),1/lamb)
        dv[j,i] = shrink((p[j,i+1]-p[j,i])/(N+1),1/lamb) #we reverse j and i since the dimensions are flipped- (N,N+1) instead of (N+1,N)

#first bs
bh=np.zeros((N+1,N))
bv=np.zeros((N,N+1))

for i in range(N+1): #again we have i refer to the long axis, not the x axis
    for j in range(N): #j is our short axis, not necessarily the y axis
        bh[i,j] = bh[i,j] -dh[i,j] + (p[i+1,j]-p[i,j])/(N+1)
        bv[j,i] = bv[j,i] -dv[j,i] + (p[i,j+1]-p[i,j])/(N+1)

for k in range(iterat):
    
    #form b
    b = (mu/(N+1)**2)*phat
    for i in range(N):
        for j in range(N):
            b[i,j] = b[i,j] 
    
    
    
    
    
    
    for i in range(N):
        for j in range(N):
            dh[i,j] = shrink(bh[i,j] + ((p[i+1,j]-p[i,j])/(N+1)),1/lamb)
            dv[i,j] = shrink(bv[i,j] + ((p[i,j+1]-p[i,j])/(N+1)),1/lamb)
            
    for i in range(N):
        for j in range(N):
            bh[i,j] = bh[i,j] -dh[i,j] + (p[i+1,j]-p[i,j])/(N+1)
            bv[i,j] = bv[i,j] -dv[i,j] + (p[i,j+1]-p[i,j])/(N+1)

