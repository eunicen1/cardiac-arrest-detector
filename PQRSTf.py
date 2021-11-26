import numpy as np
import matplotlib.pyplot as plt
#mitbih = 30 mins \ half-hour = 30*60s
#ecgiddb = 20 s 

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

#Input = 1-d signal array, time of collection in s
#Output: 2-d array of average [P-wave amplitude,
# P-wave width from 0-0,
# QRS-wave amplitude,
# QRS-wave width from 0-0,
# T-wave amplitude,
# T-wave width from 0-0,
# frequency of overall signal] = examples by IID
def detectPQRSTf(sig, time):
    n = sig.shape[0] #retrieve signal length, n
    neginds = [] #initialize negative index 
    # - peaks are usually encapsulated by negative data
    # ----- 00 ++++++ peak +++++ 00 -----
    for i in range(n):
        # append negative indexes to array 
        if(sig[i] < 0):
           neginds.append([i,sig[i]]) 
    neginds = np.array(neginds)
    if neginds.shape[0] == 0 or neginds is None:
        return 0,0,0,0,[0,0,0] 
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
        thresh = 40
        if int(percent) >= thresh:
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
    farrP = 0
    farrQRS = 0
    farrT = 0

    Pamplitude = 0
    if P.shape[0] > 0:
        Pamplitude = np.mean(P[:,1])
        #[mean frequency of P wave, mean frequency of QRS wave, mean frequency of T wave]    
        farrP = time/np.mean(np.diff(P[:,0]))
    QRSamplitude = 0
    if QRS.shape[0] > 0:
        QRSamplitude = np.mean(QRS[:,1])
        #[mean frequency of P wave, mean frequency of QRS wave, mean frequency of T wave]    
        farrQRS = time/np.mean(np.diff(QRS[:,0]))
    Tamplitude = 0
    if T.shape[0] > 0:
        Tamplitude = np.mean(T[:,1])
        #[mean frequency of P wave, mean frequency of QRS wave, mean frequency of T wave]    
        farrT = time/np.mean(np.diff(T[:,0]))
    
    fs = [farrP, farrQRS, farrT]
    plt.show()

    return Pamplitude, QRSamplitude, Tamplitude, np.mean(fs), fs
