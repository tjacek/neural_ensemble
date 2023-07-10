import argparse
import pandas as pd 
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
import statsmodels.api as sm

parser = argparse.ArgumentParser()
parser.add_argument("--stats", type=str, default='reg/RF')
args = parser.parse_args()

df=pd.read_csv(args.stats)
df['samples']=df['samples'].apply(lambda x:np.log(x))
df=df.round(2)
print(df.to_latex())

features=['classes','samples','features','gini']
X=df[features].to_numpy()
y=df['quality'].to_numpy()
X=preprocessing.normalize(X,axis=0)
mod = sm.OLS(y,X)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|'].to_numpy()
p_values=np.round(p_values,4)
norm_coeff=fii.params/np.sum(np.abs(fii.params))
norm_coeff=np.round(norm_coeff,4)

for i,feat_i in enumerate(features):
    print(f'{feat_i},{norm_coeff[i]},{p_values[i]}')

#reg=linear_model.HuberRegressor()
#reg.fit(X,y)
#norm_coeff=(reg.coef_/np.sum(np.abs(reg.coef_)))
#norm_coeff=np.round(norm_coeff,4)
#print(norm_coeff)