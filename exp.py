import numpy as np
import os 
import wfdb
import matplotlib.pyplot as plt
from wfdb import processing as prc

record_ca = wfdb.rdrecord('mitbih/420') 
record_normal = wfdb.rdrecord('ecgiddb/Person_01/rec_1')
# cdcarrest = np.array(record_ca[0])
# normal = np.array(record_normal[0])

# knorm = normal[:,0]
print(record_ca.__dict__['fs'])
print(record_ca.__dict__['p_signal'])

# print(record_normal.__dict__['fs'])
# print(record_normal.__dict__['p_signal'])

xca = record_ca.__dict__['p_signal'][:, 0]
xnorm = record_normal.__dict__['p_signal'][:, 0]
# plt.plot(xca)
# plt.plot(xnorm)
# plt.legend(['c a', 'normal'])
# plt.show()

#each PQRST = example
#PQRST has P wave, QRS wave, T wave
#P wave brings frequency, amplitude, width
#QRS wave brings frequency, amplitude, width
#T wave brings frequency, amplitude, width
#X=nx9
# sigca, fieldsca = wfdb.rdsamp('mitbih/420')
# caqrs = prc.gqrs_detect(sig=sigca[:,0], fs=fieldsca['fs'])
# pks = prc.find_peaks(sig=sigca[:,0])
# print(caqrs, pks[0])

#Input = 1nd signal array
def detectFrequency(sig):
    n = sig.shape[0]
    Δy = np.zeros(n)
    for i in range(1, n-1):
        #local maxima
        if(sig[i]-sig[i-1]) >= 0 and (sig[i]-sig[i+1]) >= 0:
            Δy[i] = sig[i]
        #local minima
        if(sig[i]-sig[i-1]) <= 0 and (sig[i]-sig[i+1]) <= 0:
            Δy[i] = sig[i]
    print("dely", Δy)

    plt.plot(sig)
    plt.annotate()

    # Δymin = np.min([ np.abs(sig[i] - sig[i-1]) for i in range(1,n) ])
    # Δymax = np.max([ np.abs(sig[i] - sig[i-1]) for i in range(1,n) ])
    # Δyavg = np.mean([ np.abs(sig[i] - sig[i-1]) for i in range(1,n) ])

    # sigmax = np.max(sig)
    # sigmin = np.min(sig)
    # print("ymax step:", Δymax)
    # print("ymin step:", Δymin)
    # print("yavg step:", Δyavg)
    # print("sigmax: ", sigmax)
    # print("sigmin:", sigmin)


detectFrequency(xca)
def pqrst(sig):
    pass

