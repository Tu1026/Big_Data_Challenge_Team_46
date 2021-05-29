from pathlib import Path
import sys
import os
import pandas as pd
import re

dir_path = Path(sys.argv[1])
df = pd.DataFrame()

for subdir, dirs, files in os.walk(dir_path):
    for dir in dirs:
        for file in files:
            if re.match(".Real.", file):
                
                

