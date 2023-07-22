import pandas as pd

df=pd.read_csv('alp.csv')
norm_dir={}
for name,values in df.iteritems():
    value_i=(values- values.mean())/values.std() 
    norm_dir[name]=value_i	

new_df=pd.DataFrame(norm_dir)
print(new_df)