import numpy as numpy
import os 
import wfdb
from IPython.display import display


record = wfdb.rdsamp('mitbih/420') 
print(record[0])