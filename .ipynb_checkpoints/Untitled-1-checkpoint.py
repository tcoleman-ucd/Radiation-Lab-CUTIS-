import matplotlib.pyplot as plt
import numpy as np
import math as m
import glob
import Data_Extraction
from pathlib import Path


here = Path(".")
Barium = here/"BGO_Cs_0_10cm_180s.Spe"
g = "BGO_Background_180s.Spe"


count_sec, channels = Data_Extraction.counts_per_sec(Barium)
count_sec1, channels = Data_Extraction.counts_per_sec(g)

def remove_background(filenameb, files):

   background_removed = filenameb- files
    
   return background_removed

vv = remove_background(count_sec, count_sec1)