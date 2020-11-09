# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 04:01:38 2019

@author: JC
CID:01063446
"""

import numpy as np
from CLA_P1_Q1 import householderQR


readings2 = np.genfromtxt('readings2.csv', delimiter =',')

R3 = householderQR(readings2)[1] #2nd output is R
print(R3)
np.allclose(R3[2:,:],np.zeros_like(R3[2:,:])) #check against zeros
