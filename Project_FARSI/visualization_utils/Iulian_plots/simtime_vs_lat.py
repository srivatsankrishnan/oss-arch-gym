#Author: Iulian Brumar
#Script that plots latency error per application

import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv(sys.argv[1])

print(data["error"])
print(data["app"])

sns.barplot(data=data, x = "app", y = "error")
plt.savefig('error_analysis.png')
