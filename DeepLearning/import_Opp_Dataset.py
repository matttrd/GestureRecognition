'''

Import Opportunity Dataset [citation needed]

'''

import pandas as pd
import numpy as np


#%%

#relative path
source_dir = r'./data/OpportunityUCIDataset/dataset/'
column_names_file = source_dir + 'column_names.txt'


with open(column_names_file) as f:
    read_data = np.array(f.read().splitlines())

idx = np.arange(2,len(read_data) - 10)    
idx2 = np.arange(len(read_data) - 7, len(read_data))
idx = np.concatenate((idx,idx2))
read_data = read_data[idx]
column_names = list(map(lambda x: x.split('Column: ')[1].split(' ',1)[1].split(';')[0], read_data))

subjects = ['S1','S2','S3','S4']
adls = ['ADL1','ADL2','ADL3','ADL4','ADL5']

labels = ['Locomotion',
          'HL_Activity',
          'LL_Left_Arm',
          'LL_Left_Arm_Object',
          'LL_Right_Arm',
          'LL_Right_Arm_Object',
          'ML_Both_Arms']

df = pd.DataFrame()
for sub in subjects:
    for adl in adls:
        tmp = pd.read_csv(source_dir + sub + '-' + adl + '.dat', sep = ' ', names = column_names)
        tmp['subject_id'] = sub
        tmp['adl_id'] = adl
        df = pd.concat([df,tmp])
        
df.reset_index(inplace = True, drop_index = True)
#%%
jacket_keys = ['Inertial']
not_used =  'SHOE'     
predictors_col = [col for col in df.columns for key in jacket_keys if key in col]
predictors_col = [col for col in predictors_col if not not_used in col]

train = ['ADL1','ADL2','ADL3']
test = ['ADL4','ADL5']

df_train = df[ df.adl_id.map(lambda x: x in train)]
df_test = df[ df.adl_id.map(lambda x: x in test)]

df_train = df_train[predictors_col + labels]
df_test = df_test[predictors_col + labels]



