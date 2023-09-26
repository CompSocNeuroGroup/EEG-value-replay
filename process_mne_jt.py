#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:39:28 2023

@author: jthompsz
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import mne
import picard

from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator, Scaler, Vectorizer, cross_val_multiscore
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
import autoreject


matplotlib.use('QtAgg')

%matplotlib qt

# set some variables
cond1    = 'Stimulus/S  1'
cond2    = 'Stimulus/S  2'
cond3    = 'Stimulus/S  3'
cond4    = 'Stimulus/S  4'
cond5    = 'Stimulus/S  5'
cond6    = 'Stimulus/S  6'
cond7    = 'Stimulus/S  7'

epochLowLim = -0.4;
epochHiLim  = 0.8;
baseline = (None, 0)

# load in data
sample_data_folder = "/home/jthompsz/data/EEG_Localizer/"
sample_data_raw_file = os.path.join(sample_data_folder, "localizer_test001.vhdr")
raw = mne.io.read_raw_brainvision(sample_data_raw_file)

# add electode montage
easycap_montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(easycap_montage)
#fig = raw.plot_sensors(show_names=True)

# rereference to common average
raw.load_data()
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels="average")

# plot to look for dud electrodes
raw_avg_ref.plot(duration=10, scalings=40e-6)
# need method to drop?
#rmChans = [0,1,2,5]
#raw_avg_ref.drop_channels(rmChans)


# ICA on raw filtered data
raw_filt = raw_avg_ref.copy().filter(l_freq=1, h_freq=30)
raw_ica = mne.preprocessing.ICA()
raw_ica.fit(raw_filt)
explained_var_ratio = raw_ica.get_explained_variance_ratio(raw_filt)

# plot some ICA diagnostics
raw_ica.plot_components(inst=raw_filt)
raw_ica.plot_sources(raw_filt)
raw_ica.plot_overlay(raw_filt, exclude=[0,1,2,5])
#raw_ica.plot_properties(raw_filt)

# plot cleaned data
reconst_raw = raw_filt.copy()
raw_ica.exclude = [0, 1,2,5]
raw_ica.apply(reconst_raw)
reconst_raw.plot()

# Apply ICA to data and epoch
raw_filt_2 = raw_avg_ref.copy().filter(l_freq=0.5, h_freq=30)
raw_ica.apply(raw_filt_2)
raw_filt_2.plot(show_scrollbars=False)
events_from_annot, event_dict = mne.events_from_annotations(raw)

# epoch data and decimate from 500Hz to 100Hz
epochs_all = mne.Epochs(raw_filt_2, events_from_annot, tmin=epochLowLim, tmax=epochHiLim, event_id=event_dict, preload=True, event_repeated='drop', baseline=baseline, decim=5)
epochs = epochs_all[cond1, cond2, cond3, cond4,cond5, cond6,cond7]

# Automated epoch rejection
# Autoreject (local) on blink-artifact-free epochs
auto_reject_epochs = autoreject.AutoReject(random_state = 100).fit(epochs[:20])
epochs_clean = auto_reject_epochs.transform(epochs)

# Decoding localizer using L1 regularized (lasso) logistic regression 
X = epochs_clean.get_data()
y = epochs_clean.events[:, 2]

clf = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', C=0.3, multi_class="ovr"))
n_splits = 5
scoring = 'accuracy'
cv = StratifiedKFold(n_splits=n_splits)
time_decoder = SlidingEstimator(clf, scoring=scoring, n_jobs=1, verbose=True)
scores = cross_val_multiscore(time_decoder, X, y, cv=5, n_jobs=4)

# Plot results of decoding
mean_acc = round(np.mean(scores), 3)
std_acc = round(np.std(scores), 3)
print(f'CV scores: {scores}')
print(f'Mean ACC = {mean_acc:.3f} (SD = {std_acc:.3f})')

fig, ax = plt.subplots()

ax.axhline(0.14, color='k', linestyle='--', label='chance')  # AUC = 0.5
ax.axvline(0, color='k', linestyle='-')  # Mark time point zero.
ax.plot(epochs.times, np.mean(scores,axis=0), label='score')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Predicted Probability')
ax.legend()
ax.set_title('Localizer - Multiclass')
fig.suptitle('Sensor Space Decoding')

