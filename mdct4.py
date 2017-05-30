
# coding: utf-8

# Implementation of MDCT
# ===
# This implementation is modelled after http://www.ee.columbia.edu/~marios/mdct/mdct4.m.  By PvdS, 2015.
# 
# The following comments are from the original author:
# 
# MDCT4 Calculates the Modified Discrete Cosine Transform
#    y = mdct4(x)
# 
#    Use either a Sine or a Kaiser-Bessel Derived window (KBDWin)with 
#    50% overlap for perfect TDAC reconstruction.
#    Remember that MDCT coefs are symmetric: y(k)=-y(N-k-1) so the full
#    matrix (N) of coefs is: yf = [y;-flipud(y)];
# 
#    x: input signal (can be either a column or frame per column)
#       length of x must be a integer multiple of 4 (each frame)     
#    y: MDCT of x (coefs are divided by sqrt(N))
# 

# In[46]:

import numpy as np


# Modified discrete cosine transform
# -----

# In[48]:

def mdct4(x):
    N = x.shape[0]
    if N%4 != 0:
        raise ValueError("MDCT4 only defined for vectors of length multiple of four.")
    M = N // 2
    N4 = N // 4
    
    rot = np.roll(x, N4)
    rot[:N4] = -rot[:N4]
    t = np.arange(0, N4)
    w = np.exp(-1j*2*np.pi*(t + 1./8.) / N)
    c = np.take(rot,2*t) - np.take(rot, N-2*t-1)         - 1j * (np.take(rot, M+2*t) - np.take(rot,M-2*t-1))
    c = (2./np.sqrt(N)) * w * np.fft.fft(0.5 * c * w, N4)
    y = np.zeros(M)
    y[2*t] = np.real(c[t])
    y[M-2*t-1] = -np.imag(c[t])
    return y


# Inverse modified discrete cosine transform
# -----

# In[49]:

def imdct4(x):
    N = x.shape[0]
    if N%2 != 0:
        raise ValueError("iMDCT4 only defined for even-length vectors.")
    M = N // 2
    N2 = N*2
    
    t = np.arange(0,M)
    w = np.exp(-1j*2*np.pi*(t + 1./8.) / N2)
    c = np.take(x,2*t) + 1j * np.take(x,N-2*t-1)
    c = 0.5 * w * c
    c = np.fft.fft(c,M)
    c = ((8 / np.sqrt(N2))*w)*c
    
    rot = np.zeros(N2)
    
    rot[2*t] = np.real(c[t])
    rot[N+2*t] = np.imag(c[t])
    
    t = np.arange(1,N2,2)
    rot[t] = -rot[N2-t-1]
    
    t = np.arange(0,3*M)
    y = np.zeros(N2)
    y[t] = rot[t+M]
    t = np.arange(3*M,N2)
    y[t] = -rot[t-3*M]
    return y


# That's it.
# Now we run it on a sample "data set".  I used the above matlab implementation to test this again.  An asine test but it is one!

# In[50]:

x = (np.arange(0.,24))


# In[51]:

y = mdct4(x)
print(y)


# In[52]:

z = imdct4(y)
print(z)


# THe following test is after https://github.com/stevengj/MDCT.jl, but that has a small error.  The idea of this test is, have two overlapping windows, do a forward then an inverse mdct, so that the errors cancel out.  The result of this test should be 0, within machine FP accuracy.

# In[53]:

X = np.random.rand(1000)
Y1 = mdct4(X[:100])
Y2 = mdct4(X[50:150])
Z1 = imdct4(Y1)
Z2 = imdct4(Y2)
print(np.linalg.norm(Z1[50:100] + Z2[:50] - 2*X[50:100])) # should be about 0


# In[ ]:



