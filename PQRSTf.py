import numpy as np
import os 
import wfdb
import matplotlib.pyplot as plt
from wfdb import processing as prc
from denoise import denoise

name_ca = 'mitbih/420'
name_norm = 'ecgiddb/Person_01/rec_1'
record_ca = wfdb.rdrecord(name_ca) 
record_norm = wfdb.rdrecord(name_norm)
print('frequency of signal '+ name_ca, ': ', record_ca.__dict__['fs'])
print('frequency of signal '+ name_norm, ': ', record_norm.__dict__['fs'])
xca = record_ca.__dict__['p_signal'][:, 0] #0or1: check which channel corresponds to what
xnorm = record_norm.__dict__['p_signal'][:, 0]
xca = denoise(xca)
xnorm = denoise(xnorm)

#Runtime below: ~O(n log n + n) = ~O(n log n)

#Input: 2-d array [indexpeak, peakamp]
#Output: 2-d array of [indexpeak, peakamp, ratiorel2Avgpeak] 
def pkRatio(arr, p=0.33):
    print("top "+str(p*100.0)+"%-peaks")
    #find mean of top p-th% peaks
    n,_ = arr.shape
    Δp = int(p*n)
    arr2 = np.copy(arr[:,1])
    arr2.sort(kind='mergesort') # sorts array in place
    # print(arr2, arr2[n-Δp-1:])
    avgppeak = np.mean(arr2[n-Δp-1:])
    #determine ratio of peaks 
    ratios = (100*arr[:,1]/avgppeak).reshape(n,1)
    return np.concatenate((arr, ratios), axis=1)

#Input = 1-d signal array
#Output: 2-d array of average [P-wave amplitude,
# P-wave width from 0-0,
# QRS-wave amplitude,
# QRS-wave width from 0-0,
# T-wave amplitude,
# T-wave width from 0-0,
# frequency of overall signal] = examples by IID
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
    rmaxs = pkRatio(maxs, p=1/(n))#, p=float((maxs.shape[0])**(-1)))
    r, _ = rmaxs.shape
    rmaxs = np.concatenate((rmaxs, (np.arange(r)).reshape(r,1)), axis=1)

    #get estimated QRS peaks
    QRS = []
    for i in range(r):
        percent = rmaxs[i,2] # obtain percentage %
        if int(percent) >= 40:
            QRS.append([rmaxs[i,0], rmaxs[i,1], i])
    QRS = np.array(QRS)
    s, _ = QRS.shape

    #get estimated P, T, QRS(mod) peaks 
    P = []
    T = []
    for i in range(s):
        start = int(QRS[i-1, 2])
        stop = int(QRS[i, 2])
        slice = rmaxs[start+1: stop, :]
        if slice.shape[0] > 1:
            if i == 0:
                P.append([slice[np.argmax(slice[:, 1]), 0], np.max(slice[:, 1])])
            elif i == s-1:
                T.append([slice[np.argmax(slice[:, 1]), 0], np.max(slice[:, 1])])
            else: 
                #predict for 2 maxes and append greater distance from QRS to P then further to T
                #T_{i-1}
                maxT = np.max(slice[:,1])
                idxmaxT = slice[np.argmax(slice[:, 1]), 0]
                rmaxidxT = slice[np.argmax(slice[:, 1]), 3]
                #P_{i}
                idxmaxP =  rmaxs[stop-1, 0]
                maxP = 0
                sliceP = rmaxs[int(rmaxidxT)+1: stop, :]
                if sliceP.shape[0] > 0:
                    maxP = np.max(sliceP[:,1])
                    idxmaxP = sliceP[np.argmax(sliceP[:, 1]), 0]
                T.append([idxmaxT, maxT])
                P.append([idxmaxP, maxP])
        else:
            if i == 0:
                p0 = (stop-start)/2
                P.append([p0, 0])
            elif i == s:
                tn = (stop-start)/2
                T.append([tn, 0])
            else:
                ti1 = start + (1/3)*int(stop - start)
                pi = start + (2/3)*int(stop - start)
                P.append([pi, 0])
                T.append([ti1, 0])
            
    P = np.array(P)
    T = np.array(T)

    Pamplitude = np.mean(P[:,1])
    QRSamplitude = np.mean(QRS[:,1])
    Tamplitude = np.mean(T[:,1])
    
    #[mean frequency of P wave, mean frequency of QRS wave, mean frequency of T wave]    
    fs = [np.mean(np.diff(P[:,0])), np.mean(np.diff(QRS[:,0])), np.mean(np.diff(T[:,0]))]
    f = n/np.mean(fs)    
    # np.savetxt("neginds.csv", neginds, delimiter=",")
    plt.plot(sig)
    plt.scatter(rmaxs[:,0], rmaxs[:,1], color='cyan')
    plt.scatter(QRS[:,0], QRS[:,1], color='maroon')
    plt.scatter(P[:,0], P[:,1], color='lime')
    plt.scatter(T[:,0], T[:,1], color='red')
    plt.show()

    return Pamplitude, QRSamplitude, Tamplitude, f, fs

print(detectPQRSTf(xca))
print(detectPQRSTf(xnorm))