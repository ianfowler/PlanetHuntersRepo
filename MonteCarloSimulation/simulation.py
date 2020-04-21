from preprocess.preprocess import *
import matplotlib.pyplot as plt
import pandas as pd
import lightkurve as lk
import time
import pickle
# from lightkurve import *
TCE_DIR = "dr24_tce_full.csv"
KEPLER_DATA_DIR = "../../kepler/"
FIG_DIM = (20, 10)
TCE_DIR = "dr24_tce_full.csv"
tce_df = pd.read_csv(TCE_DIR, skiprows=159)
cols = ["av_training_set", "tce_period", "tce_duration", "tce_time0bk", "kepid", "tce_plnt_num"]
tce_df = tce_df[cols]
tce_df = tce_df[tce_df.av_training_set.isin(["AFP", "NTP", "PC"])]
print(tce_df.head())

import batman
import lightkurve
from Engine import *

def reject_outliers(time, data, m = 3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    filt = s<m
    
    mu = np.mean(data[filt])
    sigma = np.std(data[filt])
    
    newFlux = []
    for i, flux in enumerate(data):
        if filt[i]:
            newFlux.append(flux)
        else:
            newFlux.append(np.random.normal(mu, sigma))
    
    return time, np.array(newFlux)


def scatter_lg(d, label, ax, xlbl=None, title=None):
    ax.scatter(np.arange(-0.5, 0.5, 1/len(d)), d, label=label)
    ax.set_ylabel("Normalized flux")
    if xlbl:
        ax.set_xlabel(xlbl)
    if title:
        ax.set_title(title)
    ax.legend(loc='upper right')
    
    
import lightkurve as lk
def normalize_zero_negative_one(arr):
    arr = arr - np.median(arr)
    arr = arr * (-1/np.min(arr))
    return arr

def inject_transit(kepid):

    tce = tce_df[tce_df.kepid == kepid].iloc[0]
    period = tce["tce_period"]
    duration = tce["tce_duration"]
    t0 = tce["tce_time0bk"]
    kepid = tce["kepid"]
    planet_num = tce["tce_plnt_num"]
    train_set = tce["av_training_set"]
    
    all_time, all_flux = read_light_curve(kepid, KEPLER_DATA_DIR)
    
    filtered_time, filtered_flux = [], []
    for i in range(len(all_time)):
        t, f = reject_outliers(all_time[i], all_flux[i])
        filtered_time.append(t)
        filtered_flux.append(f)
        
    time_days, flux = process_light_curve(filtered_time, filtered_flux)
    
    
    trans, orbital_period_days, inject_duration = get_transit(5700, time_days, t0)
    print(orbital_period_days, t0, inject_duration)
    

    sub_flux = flux - (trans/500)
    
    folded_time, folded_values = phase_fold_and_sort_light_curve(time_days, sub_flux, orbital_period_days, t0)
    
    
    final_flux = normalize_zero_negative_one(folded_values)
    
    
    
    g = global_view(folded_time, final_flux, orbital_period_days)
    l = local_view(folded_time, final_flux, orbital_period_days, 200)
    fig, axs = plt.subplots(2, sharex=True, figsize=FIG_DIM)
    scatter_lg(g, "Global View", axs[0], title="TCE {}-{}".format(kepid, planet_num))
    scatter_lg(l, "Local View", axs[1], xlbl="Phase-folded datapoints, centered at t0")
    plt.show()
    
# save_local_global_of_row_index(13)

# 11442793

inject_transit(11442793)

    
