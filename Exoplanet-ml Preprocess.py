#!/usr/bin/env python
# coding: utf-8

# In[176]:


from shallue_vanderburg_util.preprocess import *
import matplotlib.pyplot as plt
import pandas as pd
import lightkurve as lk
from lightkurve import *


# In[177]:


TCE_DIR = "dr24_tce_full.csv"
KEPLER_DATA_DIR = "../kepler/"
FIG_DIM = (20, 5)
# a 6022556
# b 7978202
KEPID = 6022556


# In[178]:


TCE_DIR = "dr24_tce_full.csv"
tce_df = pd.read_csv(TCE_DIR, skiprows=159)
tce = tce_df[tce_df.kepid == KEPID].iloc[0]
period = tce["tce_period"]
duration = tce["tce_duration"]
t0 = tce["tce_time0bk"]


# In[179]:


all_time, all_flux = read_light_curve(KEPID, KEPLER_DATA_DIR)


# In[180]:


time, flux = process_light_curve(all_time, all_flux)


# In[181]:


generate_example_for_tce(time, flux, tce)
