import os

from PQRSTf import detectPQRSTf
from denoise import denoise
import numpy as np
import matplotlib.pyplot as plt

# import wfdb
import pandas as pd

os.chdir('ecgiddb') #change to normal ecg; t=20s

#dictionary of P, QRS, T, f, fP, fQRS, fT
sigD = {}
# for norm in os.listdir():
#     name_norm = norm + "/rec_1"
#     record_norm = wfdb.rdrecord(name_norm)
#     xnorm = denoise(record_norm.__dict__['p_signal'][:, 0])
#     P, QRS, T, f, fs = detectPQRSTf(xnorm, 20)
#     sigD[name_norm] = [
#         round(np.float64(P),6),
#         round(np.float64(QRS),6),
#         round(np.float64(T),6),
#         round(np.float64(f),6),
#         round(np.float64(fs[0]),6),
#         round(np.float64(fs[1]),6),
#         round(np.float64(fs[2]),6),
#         round(np.float64(0),6),
#         ]

os.chdir('../cardially') #change to ca ecg; t=9s
print(os.getcwd())
#dictionary of P, QRS, T, f, fP, fQRS, fT
files = os.listdir()
for ca in files:
    tfile = open(ca, "r")
    lines = (tfile.read().splitlines())
    arr = np.float64([ line.split() for line in lines])
    arr = arr[:,1] #time approximated to 9s and x-axis removed
    xca = denoise(arr)
    P, QRS, T, f, fs = detectPQRSTf(xca, 9)
    sigD[ca] = [
        round(np.float64(P),6),
        round(np.float64(QRS),6),
        round(np.float64(T),6),
        round(np.float64(f),6),
        round(np.float64(fs[0]),6),
        round(np.float64(fs[1]),6),
        round(np.float64(fs[2]),6),
        round(np.float64(1),6),
        ]

sigPD = pd.DataFrame.from_dict(sigD).T
print(sigPD)
# colnames = ['P', 'QRS', 'T', 'f', 'fP', 'fQRS', 'fT', 'y']
# sigPD.columns = colnames
# sigPD=sigPD.fillna(0)
# sigPD.to_csv('../data.csv')

#cardiac arrest == 1
#normal == 0
