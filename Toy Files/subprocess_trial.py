import subprocess
import sys
return_code = subprocess.call("source activate tensorflow", shell=True)
print(return_code)
return_code = subprocess.call("source deactivate tensorflow", shell=True)
