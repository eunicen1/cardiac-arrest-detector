import numpy as np
import os 
import wfdb
import matplotlib.pyplot as plt
from wfdb import processing as prc

name_ca = 'mitbih/420'
name_norm = 'ecgiddb/Person_01/rec_1'
record_ca = wfdb.rdrecord(name_ca) 
record_norm = wfdb.rdrecord(name_norm)
print('frequency of signal '+ name_ca, ': ', record_ca.__dict__['fs'])
print('frequency of signal '+ name_norm, ': ', record_norm.__dict__['fs'])
xca = record_ca.__dict__['p_signal'][:, 1]
xnorm = record_norm.__dict__['p_signal'][:, 1]

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
    print(arr2, arr2[n-Δp-1:])
    avgppeak = np.mean(arr2[n-Δp-1:])
    #determine ratio of peaks 
    ratios = 100*arr[:,1]/avgppeak
    print(ratios)
    return np.concatenate((arr, ratios.reshape(n,1)), axis=1)

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
    rmaxs = pkRatio(maxs, p=0.40)#, p=float((maxs.shape[0])**(-1)))
    r, _ = rmaxs.shape

    #get estimated QRS peaks
    QRS = []
    for i in range(r):
        percent = rmaxs[i,2] # obtain percentage %
        if int(percent) >= 60:
            QRS.append([rmaxs[i,0], rmaxs[i,1], i])
    QRS = np.array(QRS)
    s, _ = QRS.shape

    #get estimated P, T, QRS(mod) peaks 
    P = []
    T = []
    QRSnew = []
    for i in range(1,s-2, 2):
        # print(i-1, i, i+1)
        startP = int(QRS[i-1,2])
        stopP = int(QRS[i,2])
        startT = int(QRS[i,2])
        stopT = int(QRS[i+1,2])
        # print(rmaxs[startP,0], rmaxs[stopP,0], rmaxs[stopT,0])
        sliceP = rmaxs[startP+1:stopP,:] #get Pzone (peaks)
        sliceT = rmaxs[startT+1:stopT,:] #get Tzone (peaks)
    
        #most significant peak (max in range between 2 QRS peaks)
        if sliceP.shape[0] > 0 and sliceT.shape[0] > 0: #should we remove QRS as well?
            QRSnew.append(QRS[i,:])
            P.append([sliceP[np.argmax(sliceP[:,1]),0], np.max(sliceP[:,1])])
            T.append([sliceT[np.argmax(sliceT[:,1]),0], np.max(sliceT[:,1])])
    
    P = np.array(P)
    T = np.array(T)
    QRS = np.array(QRSnew)

    Pamplitude = np.mean(P[:,1])
    QRSamplitude = np.mean(QRS[:,1])
    Tamplitude = np.mean(T[:,1])
    
    #[mean frequency of P wave, mean frequency of QRS wave, mean frequency of T wave]    
    fs = [np.mean(np.diff(P[:,0])), np.mean(np.diff(QRS[:,0])), np.mean(np.diff(T[:,0]))]
    f = n/np.mean(fs)
    
    np.savetxt("neginds.csv", neginds, delimiter=",")
    plt.plot(sig)
    plt.scatter(neginds[:,0], neginds[:,1], color='cyan')
    plt.scatter(QRS[:,0], QRS[:,1], color='maroon')
    plt.scatter(P[:,0], P[:,1], color='lime')
    plt.scatter(T[:,0], T[:,1], color='red')
    plt.show()

    return Pamplitude, QRSamplitude, Tamplitude, f, fs

print(detectPQRSTf(xca))
print(detectPQRSTf(xnorm))
