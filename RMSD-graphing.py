#!/usr/bin/env python
# coding: utf-8

# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[4]:


file_input = input("Filename (corresponds with axis title): ")
RMSD_data_frame = pd.read_csv(file_input, sep=',')
# practice w 7D35a_1us_rmsd_bb.csv


# In[5]:


pandas_descrip = pd.DataFrame.describe(RMSD_data_frame)
pandas_info = pd.DataFrame.info(RMSD_data_frame)
print(pandas_descrip, pandas_info)


# In[6]:


data_array = np.array(RMSD_data_frame)
print("shape of array is", data_array.shape)


# In[7]:


RMSD = data_array
cols = np.array(RMSD_data_frame.columns)

dataframe = {}
for i in range(len(cols)):
    string = cols[i]
    globals()[string] = RMSD[:,i]
    vals = globals()[string]
    dataset = dataframe.update({string: vals})
    
print(dataframe)
print(cols)


# In[8]:


time1a = dataframe['time1a']
time2a = dataframe['time2a']
time3a = dataframe['time3a']

time1c = dataframe['time1c']
time2c = dataframe['time2c']
time3c = dataframe['time3c']


# In[9]:


RMSD1a = dataframe['RMSD1a']
RMSD2a = dataframe['RMSD2a']
RMSD3a = dataframe['RMSD3a']

RMSD1c = dataframe['RMSD1c']
RMSD2c = dataframe['RMSD2c']
RMSD3c = dataframe['RMSD3c']


# In[10]:


fig, axs = plt.subplots(4, figsize=(10,15))

axs[0].set_title("RMSD CHARMM36m Forcefield")
axs[0].plot(time1c, RMSD1c, "g--", label="Replicate 1")
axs[0].plot(time2c, RMSD2c, "g--", label="Replicate 2")
axs[0].plot(time3c, RMSD3c, "g--", label="Replicate 3")
axs[0].set_ylabel("RMSD (nm)")
axs[0].set_xlabel("Time")
# plt.set_xlim()
# plt.set_ylim()
axs[0].grid(which="both", alpha=1)
axs[0].legend()

axs[1].set_title("CHARMM36m Forcefield")
axs[1].plot(time1c, RMSD1c, "g--", label="Replicate 1")
axs[1].set_ylabel("RMSD (nm)")
axs[1].set_xlabel("Time")
# plt.set_xlim()
# plt.set_ylim()
axs[1].grid(which="both", alpha=1)
axs[1].legend()

axs[2].set_title("AMBER Forcefield")
axs[2].plot(time1a, RMSD1a, "m--", label="Replicate 1")
axs[2].plot(time2a, RMSD2a, "m--", label="Replicate 2")
axs[2].plot(time3a, RMSD3a, "m--", label="Replicate 3")
axs[2].set_ylabel("RMSD (nm)")
axs[2].set_xlabel("Time")
# plt.set_xlim()
# plt.set_ylim()
axs[2].grid(which="both", alpha=1)
axs[2].legend()

axs[3].set_title("AMBER Forcefield")
axs[3].plot(time1a, RMSD1a, "m--", label="Replicate 1")
axs[3].set_ylabel("RMSD (nm)")
axs[3].set_xlabel("Time")
# plt.set_xlim()
# plt.set_ylim()
axs[3].grid(which="both", alpha=1)
axs[3].legend()

plt.savefig((str(file_input)+".png"))
plt.tight_layout()


# In[11]:


fig, axs = plt.subplots(4, figsize=(10,15))

axs[0].set_title( "CHARMM36m Forcefield")
axs[0].plot(time1c, RMSD1c, "g--", label="Replicate 1")
axs[0].plot(time2c, RMSD2c, "g--", label="Replicate 2")
axs[0].plot(time3c, RMSD3c, "g--", label="Replicate 3")
axs[0].set_ylabel("RMSD (nm)")
axs[0].set_xlabel("Time")
# plt.set_xlim()
# plt.set_ylim()
axs[0].grid(which="both", alpha=1)
axs[0].legend()

axs[1].set_title( "CHARMM36m Forcefield")
axs[1].plot(time1c, RMSD1c, "g--", label="Replicate 1")
axs[1].set_ylabel("RMSD (nm)")
axs[1].set_xlabel("Time")
# plt.set_xlim()
# plt.set_ylim()
axs[1].grid(which="both", alpha=1)
axs[1].legend()

axs[2].set_title("AMBER Forcefield")
axs[2].plot(time1a, RMSD1a, "m--", label="Replicate 1")
axs[2].plot(time2a, RMSD2a, "m--", label="Replicate 2")
axs[2].plot(time3a, RMSD3a, "m--", label="Replicate 3")
axs[2].set_ylabel("RMSD (nm)")
axs[2].set_xlabel("Time")
# plt.set_xlim()
# plt.set_ylim()
axs[2].grid(which="both", alpha=1)
axs[2].legend()

axs[3].set_title( "AMBER Forcefield")
axs[3].plot(time1a, RMSD1a, "m--", label="Replicate 1")
axs[3].set_ylabel("RMSD (nm)")
axs[3].set_xlabel("Time")
# plt.set_xlim()
# plt.set_ylim()
axs[3].grid(which="both", alpha=1)
axs[3].legend()

plt.savefig((str(file_input)+".png"))
plt.tight_layout()


# In[12]:


print("Finished creating noisy RMSD graphs")


# In[13]:


# test_df = {'RMSD1_charmm': md1c}
# test_df = pd.DataFrame.from_dict(data=test_df)

# test_df['moving-avg_RMSD1_charmm'] = test_df.RMSD1_charmm.rolling(window=10).mean()
# test1 = test_df['moving-avg_RMSD1_charmm']
# print(test1.head())
# test_df.plot()

### Moving Average Data 
##CHARMM 
RMSD1_charmm_dict = {'RMSD1c': RMSD1c}
RMSD1_charmm_df = pd.DataFrame.from_dict(data=RMSD1_charmm_dict)
RMSD1_charmm_df['moving_avg_RMSD1_charmm'] = RMSD1_charmm_df.RMSD1c.rolling(window=500).mean()

RMSD2_charmm_dict = {'RMSD2c': RMSD2c}
RMSD2_charmm_df = pd.DataFrame.from_dict(data=RMSD2_charmm_dict)
RMSD2_charmm_df['moving_avg_RMSD2_charmm'] = RMSD2_charmm_df.RMSD2c.rolling(window=500).mean()

RMSD3_charmm_dict = {'RMSD3c': RMSD3c}
RMSD3_charmm_df = pd.DataFrame.from_dict(data=RMSD3_charmm_dict)
RMSD3_charmm_df['moving_avg_RMSD3_charmm'] = RMSD3_charmm_df.RMSD3c.rolling(window=500).mean()

#AMBER
RMSD1_amber_dict = {'RMSD1a': RMSD1a}
RMSD1_amber_df = pd.DataFrame.from_dict(data=RMSD1_amber_dict)
RMSD1_amber_df['moving_avg_RMSD1_amber'] = RMSD1_amber_df.RMSD1a.rolling(window=500).mean()

RMSD2_amber_dict = {'RMSD2a': RMSD2a}
RMSD2_amber_df = pd.DataFrame.from_dict(data=RMSD2_amber_dict)
RMSD2_amber_df['moving_avg_RMSD2_amber'] = RMSD2_amber_df.RMSD2a.rolling(window=500).mean()

RMSD3_amber_dict = {'RMSD3a': RMSD3a}
RMSD3_amber_df = pd.DataFrame.from_dict(data=RMSD3_amber_dict)
RMSD3_amber_df['moving_avg_RMSD3_amber'] = RMSD3_amber_df.RMSD3a.rolling(window=500).mean()


# In[14]:


print("Finished creating moving average RMSD data")


# In[15]:


frames = [RMSD1_charmm_df['moving_avg_RMSD1_charmm'], RMSD1_amber_df['moving_avg_RMSD1_amber']]

result = pd.concat(frames)


# In[16]:


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

# plt.xlim([70000,100000])
RMSD1_charmm_df.plot(ax=axs[0,0], xlim=(70000,100000))
RMSD1_amber_df.plot(ax=axs[0,1], xlim=(70000,100000))

RMSD2_charmm_df.plot(ax=axs[1,0], xlim=(70000,100000))
RMSD2_amber_df.plot(ax=axs[1,1], xlim=(70000,100000))

RMSD3_charmm_df.plot(ax=axs[2,0], xlim=(70000,100000))
RMSD3_amber_df.plot(ax=axs[2,1], xlim=(70000,100000))

plt.savefig((str(file_input)+"stacked-mvavg"+".png"))
plt.tight_layout()


# In[17]:


RMSD1_charmm_df['moving_avg_RMSD1_charmm'].shape


# In[18]:


fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15,15))

RMSD1_charmm_df['moving_avg_RMSD1_charmm'].plot(ax=axs[0], color = "orange", xlim=(30000,100000))
RMSD2_charmm_df['moving_avg_RMSD2_charmm'].plot(ax=axs[0], color = "orange", xlim=(30000,100000))
RMSD3_charmm_df['moving_avg_RMSD3_charmm'].plot(ax=axs[0], color = "orange", xlim=(30000,100000))
axs[0].grid()

RMSD1_amber_df['moving_avg_RMSD1_amber'].plot(ax=axs[1], color = "blue", xlim=(30000,100000))
RMSD2_amber_df['moving_avg_RMSD2_amber'].plot(ax=axs[1], color = "blue", xlim=(30000,100000))
RMSD3_amber_df['moving_avg_RMSD3_amber'].plot(ax=axs[1], color = "blue", xlim=(30000,100000))
axs[1].grid()

RMSD1_amber_df['moving_avg_RMSD1_amber'].plot(ax=axs[2], color = "blue", xlim=(30000,100000))
RMSD2_amber_df['moving_avg_RMSD2_amber'].plot(ax=axs[2], color = "blue", xlim=(30000,100000))
RMSD3_amber_df['moving_avg_RMSD3_amber'].plot(ax=axs[2], color = "blue", xlim=(30000,100000))
RMSD1_charmm_df['moving_avg_RMSD1_charmm'].plot(ax=axs[2], color = "orange", xlim=(30000,100000))
RMSD2_charmm_df['moving_avg_RMSD2_charmm'].plot(ax=axs[2], color = "orange", xlim=(30000,100000))
RMSD3_charmm_df['moving_avg_RMSD3_charmm'].plot(ax=axs[2], color = "orange", xlim=(30000,100000))
axs[2].grid()

plt.savefig((str(file_input)+"ff-mvavg"+".png"))
plt.tight_layout()


# In[19]:


print("Finished creating Moving Average RMSD graphs")


# In[20]:


RMSD1_charmm_df.head()


# In[157]:


def hist_avg(df):
    x, y = np.histogram(np.array(df))
#     return(np.average(y))
    y_coord = []
    x_coord = []
    for i in range(2):
        x_coord.append(np.average(y))
        y_coord.append(i*max(y)*max(x))
    
    x , y = x_coord, y_coord
        
    return x, y


# In[158]:


RMSD1c_hist_avg = np.array(hist_avg(RMSD1_charmm_df["RMSD1c"]))
RMSD1a_hist_avg = np.array(hist_avg(RMSD1_amber_df["RMSD1a"]))

RMSD2c_hist_avg = np.array(hist_avg(RMSD2_charmm_df["RMSD2c"]))
RMSD2a_hist_avg = np.array(hist_avg(RMSD2_amber_df["RMSD2a"]))

RMSD3c_hist_avg = np.array(hist_avg(RMSD3_charmm_df["RMSD3c"]))
RMSD3a_hist_avg = np.array(hist_avg(RMSD3_amber_df["RMSD3a"]))


# print(RMSD1c_hist_avg[0], RMSD1c_hist_avg[1])


# In[159]:


RMSD_charmm = [RMSD1_charmm_df['RMSD1c'], RMSD2_charmm_df['RMSD2c'], RMSD3_charmm_df['RMSD3c']]
RMSD_charmm = pd.concat(RMSD_charmm, axis=0)

RMSD_amber = [RMSD1_amber_df['RMSD1a'], RMSD2_amber_df['RMSD2a'], RMSD3_amber_df['RMSD3a']]
RMSD_amber = pd.concat(RMSD_amber, axis=0)


# In[160]:


RMSD_charmm = RMSD_charmm.reset_index()
RMSD_amber = RMSD_amber.reset_index()

RMSD_charmm = RMSD_charmm.rename(columns={"index": "time", 0:'RMSD'})
RMSD_amber = RMSD_amber.rename(columns={"index": "time", 0:'RMSD'})
RMSD_amber


# In[161]:


RMSD_charmm_hist_avg = np.array(hist_avg(RMSD_charmm["RMSD"]))
RMSD_amber_hist_avg = np.array(hist_avg(RMSD_amber["RMSD"]))


# In[167]:


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15,15))

RMSD_amber['RMSD'].plot.hist(ax=axs[0],bins=50, alpha=0.5, color= "blue", label="AMBER RMSD")
RMSD_charmm['RMSD'].plot.hist(ax=axs[0],bins=50, alpha=0.75, color= "orange", label="CHARMM RMSD")
axs[0].plot(RMSD_charmm_hist_avg[0], RMSD_charmm_hist_avg[1], color="red", label="CHARMM average")
axs[0].legend()
axs[0].grid()

RMSD_charmm['RMSD'].plot.hist(ax=axs[1],bins=50, alpha=0.5, color= "orange", label="CHARMM RMSD")
RMSD_amber['RMSD'].plot.hist(ax=axs[1],bins=50, alpha=0.75, color= "blue", label="AMBER RMSD")
axs[1].plot(RMSD_amber_hist_avg[0], RMSD_amber_hist_avg[1], color="red", label="AMBER average")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.savefig((str(file_input)+"ff-mvavg-hist"+".png"))


# In[163]:


fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15,15))

RMSD1_charmm_df["RMSD1c"].plot.hist(ax=axs[0,0],bins=40, color= "orange")
axs[0,0].plot(RMSD1c_hist_avg[0], RMSD1c_hist_avg[1], color="red", label="average")
axs[0,0].legend()
RMSD1_amber_df["RMSD1a"].plot.hist(ax=axs[0,1], bins=40, color= "blue")
axs[0,1].plot(RMSD1a_hist_avg[0], RMSD1a_hist_avg[1], color="red", label="average")
axs[0,1].legend()

RMSD2_charmm_df["RMSD2c"].plot.hist(ax=axs[1,0], bins=40, color= "orange")
axs[1,0].plot(RMSD2c_hist_avg[0], RMSD2c_hist_avg[1], color="red", label="average")
axs[1,0].legend()
RMSD2_amber_df["RMSD2a"].plot.hist(ax=axs[1,1], bins=40, color= "blue")
axs[1,1].plot(RMSD2a_hist_avg[0], RMSD2a_hist_avg[1], color="red", label="average")
axs[1,1].legend()

RMSD3_charmm_df["RMSD3c"].plot.hist(ax=axs[2,0],bins=40, color= "orange")
axs[2,0].plot(RMSD3c_hist_avg[0], RMSD3c_hist_avg[1], color="red", label="average")
axs[2,0].legend()
RMSD3_amber_df["RMSD3a"].plot.hist(ax=axs[2,1], bins=40, color= "blue")
axs[2,1].plot(RMSD3a_hist_avg[0], RMSD3a_hist_avg[1], color="red", label="average")
axs[2,1].legend()


## combined RMSD hists
RMSD_charmm['RMSD'].plot.hist(ax=axs[3,0],bins=50, color= "orange", label="CHARMM RMSD")
axs[3,0].plot(RMSD_charmm_hist_avg[0], RMSD_charmm_hist_avg[1], color="red", label="CHARMM average")
axs[3,0].legend()

RMSD_amber['RMSD'].plot.hist(ax=axs[3,1],bins=50, color= "blue", label="AMBER RMSD")
axs[3,1].plot(RMSD_amber_hist_avg[0], RMSD_amber_hist_avg[1], color="red", label="AMBER average")
axs[3,1].legend()

plt.savefig((str(file_input)+"stacked-mvavg-hist"+".png"))
plt.tight_layout()


# In[165]:


print("Finished creating Moving Average RMSD histograms")

