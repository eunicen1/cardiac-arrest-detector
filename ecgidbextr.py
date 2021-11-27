import os 
import wfdb
import numpy as np 

os.chdir('ecgiddb')
files = os.listdir()
for norm in files:
    name_norm = norm + "/rec_1"
    record_norm = wfdb.rdrecord(name_norm)
    xnorm = np.array(record_norm.__dict__['p_signal'][:, 0]) #raw signal
    x = xnorm.shape[0]
    xnorm = xnorm.reshape(x,1)
    os.chdir('../ecgiddb2')
    tfile = open(norm+".txt", "w") #assume rec_1
    for row in xnorm:
        np.savetxt(tfile, row)
    tfile.close()
    os.chdir('../ecgiddb')