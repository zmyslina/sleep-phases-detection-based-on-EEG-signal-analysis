# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:21:11 2024

@author: 48511
"""


import os
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore", category=FutureWarning)

VBS = True
data_directory = "C:/Users/48511/Desktop/data_folder1"

X_data = []
y_data = []

for file_name in os.listdir(data_directory):
    if file_name.endswith("PSG.edf"):
        eeg_file_path = os.path.join(data_directory, file_name)

        raw_train = mne.io.read_raw_edf(eeg_file_path, preload=True)

        raw_train.filter(l_freq=0.5, h_freq=30, method='iir', iir_params={'order': 5, 'ftype': 'butter', 'output': 'sos'})

        hypnogram_file_name = file_name.replace("PSG.edf", "Hypnogram.edf")
        hypnogram_file_path = os.path.join(data_directory, hypnogram_file_name)

        if os.path.exists(hypnogram_file_path):
            try:
                hypnogram = np.genfromtxt(hypnogram_file_path, dtype=int, invalid_raise=False)
            except Exception as e:
                print(f"Błąd podczas wczytywania hypnogramu dla {file_name}: {e}")
                continue

            annot = mne.read_annotations(hypnogram_file_path)
            annot.crop(annot[1]["onset"] - 30 * 60, annot[-2]["onset"] + 30 * 60)
            raw_train.set_annotations(annot, emit_warning=False)

            annotation_desc_2_event_id = {
                'Sleep stage R': -1,
                'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage 4': 4
            }
            events_train, _ = mne.events_from_annotations(raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.0)

            tmax = 30.0 - 1.0 / raw_train.info["sfreq"] 
            epochs_train = mne.Epochs(
                raw=raw_train,
                events=events_train,
                event_id=annotation_desc_2_event_id,
                tmin=0.0,
                tmax=tmax,
                baseline=None,
            )
            del raw_train

            channels_of_interest = ['EEG Fpz-Cz', 'EEG Pz-Oz']

            epochs_train.load_data()

            epochs_train.pick_channels(channels_of_interest)

            psds_selected_channels, freqs = mne.time_frequency.psd_array_welch(
                epochs_train.get_data(), sfreq=epochs_train.info['sfreq'], fmin=0.5, fmax=30, n_fft=1000)

            X = psds_selected_channels.reshape(len(psds_selected_channels), -1)  
            y = np.concatenate([epochs_train.events[:, 2]]) 

            X_data.append(X)
            y_data.append(y)

        else:
            print(f"Brak pliku hypnogramu dla {file_name} ({hypnogram_file_path})")

X_data = np.concatenate(X_data)
y_data = np.concatenate(y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

svm_classifier = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, 
shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,verbose=False,
max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)

svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



unique_labels = np.unique(np.concatenate((y_test, y_pred)))
print(f"Unikalne etykiety: {unique_labels}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, ax=ax, cmap='coolwarm', fmt='g')

ax.set_xlabel('Przewidziane etykiety')
ax.set_ylabel('Prawdziwe etykiety')
ax.set_title('Macierz pomyłek')
ax.xaxis.set_ticklabels(unique_labels)
ax.yaxis.set_ticklabels(unique_labels)

plt.show()

report = classification_report(y_test, y_pred)
print("Raport klasyfikacji:\n", report)
