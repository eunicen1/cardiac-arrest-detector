import numpy as np
import os 
import wfdb
import matplotlib.pyplot as plt
from wfdb import processing as prc

record_ca = wfdb.rdrecord('mitbih/420') 
record_normal = wfdb.rdrecord('ecgiddb/Person_01/rec_1')
print(record_ca.__dict__['fs'])
print(record_ca.__dict__['p_signal'])
xca = record_ca.__dict__['p_signal'][:, 1]
xnorm = record_normal.__dict__['p_signal'][:, 1]

#Input: 2-d array [indexpeak, peakamplitude]
#Output: 2-d array of [indexpeak, peakamplitude, ratiorelativeAvgpeak] 
def pkRatio(arr, p=0.33):
    #find mean of top p-th% peaks
    n,_ = arr.shape
    Δp = int(p*n)
    arr2 = arr[:,1]
    np.sort(arr2, kind='mergesort')
    avgppeak = np.mean(arr2[n-Δp:])
    #determine ratio of peaks 
    ratios = 100*arr[:,1]/avgppeak
    return np.concatenate((arr, ratios.reshape(n,1)), axis=1)



#Input = 1-d signal array
#Output: 2-d array of estimated [P-wave amplitude,
# P-wave width from 0-0,
# QRS-wave amplitude,
# QRS-wave width from 0-0,
# T-wave amplitude,
# T-wave width from 0-0,
# avg frequency of overall signal] = examples
def detectPQRSTf(sig):
    n = sig.shape[0] #retrieve signal length, n
    neginds = [] #initialize negative index 
    # - peaks are usually encapsulated by negative data
    # ----- 00 ++++++ peak +++++ 00 -----
    for i in range(n):
        # append negative indexes to array 
        if(sig[i] < 0):
           neginds.append([i,sig[i]]) 
    neginds = np.array(neginds)
    p, _ = neginds.shape
    maxs = []
    for i in range(p-1):
        start = neginds[i,0] 
        stop = neginds[i+1,0]
        Δ = stop - start
        if Δ > 1: 
            idx = range(int(start), int(stop))
            slice = sig[int(start):int(stop)]
            #get max and index of a potential peak
            maxs.append([idx[np.argmax(slice)], np.max(slice)])
    maxs = np.array(maxs)

    # all maxs and relative closeness to max
    rmaxs = pkRatio(maxs, p=2/(maxs.shape[0]))
    r, _ = rmaxs.shape
    QRS = []
    for i in range(r):
        percent = rmaxs[i,2] # obtain percentage %
        if int(percent) >= 100:
            QRS.append([rmaxs[i,0], rmaxs[i,1]])
    QRS = np.array(QRS)

    ##[mean frequency of P-wave, mean frequency of QRS wave, mean frequency of T wave]    
    #fs = [np.mean(np.diff(P[:,0])), np.mean(np.diff(QRS[:,0])), np.mean(np.diff(T[:,0]))]
    #print(fs)
    # f = np.mean(fs) #?
    
    np.savetxt("neginds.csv", neginds, delimiter=",")
    plt.plot(sig)
    plt.scatter(neginds[:,0], neginds[:,1], color='cyan')
    plt.scatter(maxs[:,0], maxs[:,1], color='coral')
    plt.scatter(QRS[:,0], QRS[:,1], color='maroon')
    plt.show()
    

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


detectPQRSTf(xca)
detectPQRSTf(xnorm)
