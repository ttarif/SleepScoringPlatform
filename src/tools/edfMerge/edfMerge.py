import os
import tkinter as tk
from tkinter import filedialog
from tkinter.simpledialog import askstring
import mne
import numpy as np
import pyedflib

def read_edf_file(directory, edf_file):
    file_path = os.path.join(directory, edf_file)
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None, None
    file = mne.io.read_raw_edf(file_path, preload=True)
    return file.get_data(), file.info

def combine_edf_files(directory, selected_files, combined_filename):
    if not selected_files:
        print("No EDF files selected.")
        return

    # Check if combined file already exists
    output_filename = os.path.join(directory, combined_filename)
    if os.path.exists(output_filename):
        # Load existing combined data
        existing_data, existing_info = read_edf_file(directory, combined_filename)
        if existing_data is None or existing_info is None:
            print(f"Error: Failed to load existing combined file: {output_filename}")
            return
        data_list = [existing_data]
        info_list = [existing_info]
    else:
        data_list = []
        info_list = []

    # Read the data and info of the selected EDF files
    for edf_file in selected_files:
        data, info = read_edf_file(directory, edf_file)
        if data is not None and info is not None:
            data_list.append(data)
            info_list.append(info)

    # Ensure all selected files have the same channel configurations
    for info in info_list[1:]:
        if 'ch_names' not in info or 'sfreq' not in info:
            print(f"Error: Missing required information in the header of file: {edf_file}")
            return

    # Concatenate the data and merge the info dictionaries
    combined_data = np.concatenate(data_list, axis=1)
    combined_info = mne.create_info(ch_names=info_list[0]['ch_names'], sfreq=info_list[0]['sfreq'], ch_types=['eeg'] * len(info_list[0]['ch_names']))

    # Save the combined data to a new EDF file using pyedflib
    file_handle = pyedflib.EdfWriter(output_filename, len(combined_info['ch_names']), file_type=pyedflib.FILETYPE_EDFPLUS)
    file_handle.setSignalHeaders(combined_info['chs'])
    file_handle.writeSamples(combined_data)
    file_handle.close()

    print(f'Done. Combined data saved as: {output_filename}')

def browse_directory_and_select_files():
    directory = filedialog.askdirectory()
    selected_files = filedialog.askopenfilenames(initialdir=directory, title="Select EDF files", filetypes=(("EDF files", "*.edf"), ("all files", "*.*")))
    combined_filename = askstring("Combine EDF Files", "Enter a name for the combined EDF file (without extension):")
    if combined_filename:
        combine_edf_files(directory, selected_files, f"{combined_filename}.edf")

root = tk.Tk()
button = tk.Button(root, text="Select Directory and Files", command=browse_directory_and_select_files)
button.pack()
root.mainloop()
