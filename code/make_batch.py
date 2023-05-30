import datetime
import pandas as pd
import numpy as np

# for p in np.linspace(0.1, 0.9, 9):
# for p in np.linspace(0.1, 0.9, 1):
options = np.linspace(0.0, 1.0, 11)

for i in options:
    for j in options:
        # fix above print but limit decimals to two places
        print("python simulate-inter.py", 1, round(i, 2), 1, round(j, 2), ">> output.csv")
