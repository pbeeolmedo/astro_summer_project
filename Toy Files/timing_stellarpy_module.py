import glob
from stellarpy import Star

fits_folder = "../Data_Files/Spectra"
MAX_NUM_FILES = len(glob.glob(f"{fits_folder}/*.fits"))

star = Star(glob.glob(f"{fits_folder}/*.fits")[0])
subclass = star.subclass
chi_sq = star.chi_sq
