import numpy as np
import os 
import wfdb

record = wfdb.rdsamp('mitbih/420') 
record2 = wfdb.rdsamp('ecgiddb/Person_01/rec_1')
print(record[0])