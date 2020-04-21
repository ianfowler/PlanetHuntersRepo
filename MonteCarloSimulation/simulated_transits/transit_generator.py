#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import csv 
import random
import math
import os
from temp_bins.BinFinder import *
import pickle
from transit_model import generic_transit_model, compute_duration

STELLAR_RADIUS = 6.957*(10**8)

#with open('temperature_bins.obj', 'rb') as f:
#    t_bins = pickle.load(f)


def get_distribution_parameters():
    temp_distribution = pd.DataFrame()
    old_bins = [min(t_bins[0])]
    max_t = max([max(x) for x in t_bins])
    
    for i, b in enumerate(t_bins):
        old_bins.append(max(b))
        current_bin = pd.Series({
            "lower":min(b),
            "frequency":len(b)
        }, name=i)

        temp_distribution = temp_distribution.append(current_bin)
    temp_distribution["upper"] = temp_distribution["lower"].shift(-1)
    temp_distribution["upper"].iloc[-1] = max_t
    temp_distribution["scaled_frequency"] = temp_distribution["frequency"] / max(temp_distribution["frequency"])
    
    return old_bins, min([min(x) for x in t_bins]), max_t, temp_distribution

old_bins, min_temp, max_temp, temp_distribution = get_distribution_parameters()





#Inner orbital radius of habitable zone
def roi(temp):
    return (0.62817*temp**3)-(1235.15*temp**2)

#Outer orbital radius
def roo(temp):
    return (1.52*temp**3)-(2988.75*temp**2)

def starRadius(temp):
    return (temp*1.8395*10**5)-3.6169*10**8

def starMass(temp):
    return (2.85187*10**22*temp**2)+(3.70772*10**26*temp)-9.76855*10**29

def transitTime(starRadius,randOrbital,starMass):
    return (2*starRadius*math.sqrt((randOrbital*10**11)/(starMass*6.67)))

def transitDepth(planetRadius,starRadius):
    return (planetRadius**2)/(starRadius**2)

#def orbitalPeriod(randOrbital,starMass):
#    return (2*math.pi*randOrbital**1.5)*math.sqrt((randOrbital*10**11)/(starMass*6.67))

# Kepler's Third Law, returns orbital period (seconds)
def orbitalPeriod(orbital_radius,star_mass):
    GRAVITATIONAL_CONSTANT = 6.674 * (10 ** -11)
    return 2 * math.pi * (orbital_radius ** 1.5) / ((GRAVITATIONAL_CONSTANT * star_mass) ** 0.5)

def oradius_range(midTemp, steps):
    roi_, roo_ = roi(midTemp), roo(midTemp)
    return np.linspace(min(roi_, roo_), max(roi_, roo_), steps)


def choose_oradius(temp):
    o_range = oradius_range(temp, 2)
    return np.random.uniform(low=o_range[0], high=o_range[-1])

def pradius_range(midTemps, steps):
    min_planet_pradius = 3390*10**3 # Radius of Mars
    max_planet_pradius = 11467*10**3 # Radius of 1.8 Earth (came out of kepler paper)
    return np.linspace(min_planet_pradius, max_planet_pradius, steps)


def choose_pradius(temp):
    p_range = pradius_range(temp, 2)
    return np.random.uniform(low=p_range[0], high=p_range[-1])



def get_scaled_frequency(index):
    row = temp_distribution.iloc[index]
    freq = row["scaled_frequency"]
    high = row["upper"]
    low = row["lower"]
    return freq, high, low

def choose_temp():
    x = int(np.random.uniform(low=0, high=len(old_bins)-1))
    freq, high, low = get_scaled_frequency(x)
    if np.random.rand() < freq:
        return np.random.uniform(low=low, high=high)
        
    return choose_temp()
    
    

def get_transit_parameters(temp):
    star_mass = starMass(temp) # Mass of star, kg
    planet_radius = choose_pradius(temp)
    orbital_radius = choose_oradius(temp)
    orbital_period = orbitalPeriod(orbital_radius, star_mass)/(60*60*24*365.25)
    
    planet_radius_sr = planet_radius/STELLAR_RADIUS # Planet radius, stellar radii
    orbital_radius_sr = orbital_radius/STELLAR_RADIUS # Orbital radius, stellar radii
    
    
    print("orbital_period {}".format(orbital_period))
    print("planet_radius_sr {}".format(planet_radius_sr))
    print("orbital_radius_sr {}".format(orbital_radius_sr))
    
#    # time, t0, period, duration, planet_radius, impact


#
#    duration = compute_duration(orbital_period, planet_radius, orbital_radius, 90)


    duration = 14*33/365.25 # 30 minutes, converted to years
    KEPLER_CADENCE_DURATION = 30 / (60*24*365.25) # 30 minutes, converted to years
    samples = duration / KEPLER_CADENCE_DURATION
    print(duration, samples)
    
    
    
    
    
    t = np.linspace(-1*duration, duration, int(samples))

    flux = generic_transit_model(t, orbital_period, orbital_radius, planet_radius)
##    transit_model_quad(t, 0, 1, 0.02, 0.1, 0, 0.4, 0.25)
#
#
#    t = np.linspace(-1.0, 1.0, 1000)
#    m = batman.TransitModel(params, t)
    plt.scatter(t, flux)
    plt.show()
    
get_transit_parameters(5700)
