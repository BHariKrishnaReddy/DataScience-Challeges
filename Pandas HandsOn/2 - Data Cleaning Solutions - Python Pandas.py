#Write your code here
import pandas as pd
import numpy as np
height_A = pd.Series([176.2,158.4,167.6,156.2,161.4])
height_A.index = ['s1','s2','s3','s4','s5']
weight_A = pd.Series([85.1,90.2,76.8,80.4,78.9])
weight_A.index = ['s1','s2','s3','s4','s5']
df_A = pd.DataFrame()
df_A['Student_height'] = height_A
df_A['Student_weight'] = weight_A

df_A.loc['s3'] = np.nan
df_A.loc['s5'][1] = np.nan

df_A2 = df_A.dropna(how = 'any')
print(df_A2)
