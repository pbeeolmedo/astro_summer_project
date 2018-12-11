import pandas as pd
import re
df = pd.read_csv("segue_dataquery.csv") # dataframe =>df
speclist = open('speclist_urlext.txt','w')
#names = df.columns.values
#print(names)
plate = df['plate']
mjd = df['mjd']
fiberid = df['fiberid']
spectra_info = set()

for i in range(len(plate)):
    if re.match("3.*",str(plate[i])):
        spectra_info.add(str(plate[i])+"/spec-"+str(plate[i])+"-"+str(mjd[i])+"-"+str(format(fiberid[i],"04"))+".fits"+"\n")

[speclist.write(line) for line in spectra_info]
