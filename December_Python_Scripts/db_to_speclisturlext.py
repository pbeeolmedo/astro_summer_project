import pandas as pd
import re
df = pd.read_csv("../Data_Files/A0pStar_Data.csv") # dataframe =>df
speclist = open('speclist_a0pstars.txt','w')
#names = df.columns.values
#print(names)
plate = df['plate']
mjd = df['mjd']
fiberid = df['fiberid']
spectra_info = set()

for i in range(len(plate)):
    spectra_info.add(str(plate[i])+"/spec-"+str(plate[i])+"-"+str(mjd[i])+"-"+str(format(fiberid[i],"04"))+".fits"+"\n")

[speclist.write(line) for line in spectra_info]
