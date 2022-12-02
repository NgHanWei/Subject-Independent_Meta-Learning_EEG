# -*- coding: utf-8 -*-

"""
@author: Nicolás Nieto - nnieto@sinc.unl.edu.ar

Event Related Potentials - Plotting
"""

# In[] Imports modules
import mne

import numpy as np
import matplotlib.pyplot as plt

from Data_extractions import Extract_block_data_from_subject, Extract_data_from_subject
from Data_processing import Filter_by_condition, Filter_by_class
from Utilitys import Ensure_dir, picks_from_channels

# In[] Imports modules

# Root where the data are stored
root_dir = "../"
save_dir = "/media/nnieto/50e793fa-4717-4997-b590-b4167be63ee7/home/nnieto/Escritorio/Nico/Doctorado/Tesis/LIAA - sinc(i)/Toma de datos/Analisis/ERPs/Dataset final/ERPs/"

# Subjets
N_S_list=[1,2,3,4,5,6,7,8,9,10]

# Data Parameters
datatype="EEG"
Condition_list=["Pron"]
Classes_list=[ "All"]
channels = "all"

# Saving Parameters
save_bool = True
prefix="ERPs"

#Fix all random states
random_state = 23
np.random.seed(random_state)

#Plotting
# Time Windows
t_start = -0.5
t_end = 4
spatial_colors = True
plot_cues_bool = True
clim = dict(eeg=[-10, 15])
ylim = clim
bandwidth = 2
fontsize = 20
plt.rcParams.update({'font.size': fontsize})
plt.rcParams.update({'legend.framealpha':0})

# In[]: Load Data
N_B = 1
N_S = 1

# Load a single subject to use the Epoched Object structure
X_S , Y = Extract_block_data_from_subject(root_dir,N_S,datatype,N_B=N_B)

Adquisition_eq="biosemi128"
montage = mne.channels.make_standard_montage(Adquisition_eq)
X_S.set_montage(montage)

# Get picks for the selected channels
picks = picks_from_channels(channels)

n_test = 1
for Classes in Classes_list: 
    for Cond in Condition_list:
        count=1
        for N_S in N_S_list:

            # Load full subject's data
            X, Y = Extract_data_from_subject(root_dir, N_S, datatype)
        
            # Filter by condition
            X_cond , Y_cond = Filter_by_condition(X, Y, Condition = Cond)
        
            # Filter by class
            X_class , Y_class =  Filter_by_class(X_cond,Y_cond,Class=Classes)
            
            if count==1:  
                X_data = X_class
                Y_data = Y_class
                count=2
            else:
                X_data = np.vstack((X_data,X_class))
                Y_data = np.vstack((Y_data,Y_class))

        # In[]: Plotting     
        print("Ploting ERPs for Class: " + Classes +" in Condition: " + Cond )
        print("with the information of Subjects: " + str(N_S_list ))        
        
        # Put all data
        X_S._data=X_data
        X_S.events=Y_data
        X_averaged=X_S.average()

        fig= plt.figure(figsize=(20,10))
        axs= fig.add_axes([0.1,0.1,0.8,0.8])
        
        # Plot Cues
        if plot_cues_bool:
            plt.plot([0,0],[-15,15],axes=axs,color='black')
            plt.plot([0.5,0.5],[-15,15],axes=axs,color='black')
            plt.plot([3,3],[-15,15],axes=axs,color='black')
        
        # Plot ERPs
        X_averaged.plot(spatial_colors = spatial_colors, picks = picks, ylim = ylim, axes = axs, xlim= [t_start , t_end])

        title = "ERPs - Condition: " + Cond + " in Class" + Classes
        axs.set_title(title, fontsize = fontsize)
        
        # Save Figure
        if save_bool:
            Ensure_dir(save_dir)
            fig.savefig(save_dir + prefix + '_' + Cond + '_' + Classes + '_' + channels + '_.png', transparent = True)   
        
