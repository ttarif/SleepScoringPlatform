import os
import pyedflib
import matplotlib.pyplot as plt
import numpy as np

def visualize_edf(file_path, output_dir):
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        f = pyedflib.EdfReader(file_path)
        num_signals = f.signals_in_file

        # Read each signal and its properties
        signals = [f.readSignal(i) for i in range(num_signals)]
        signal_labels = f.getSignalLabels()
        sample_frequencies = f.getSampleFrequencies()

        # Create time axis
        duration = f.file_duration
        time_axis = np.arange(0, duration, 1/sample_frequencies[0])

        # Plot each signal and save the plot
        for i in range(num_signals):
            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, signals[i], label=signal_labels[i])
            plt.title("Signal: " + signal_labels[i])
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{output_dir}/signal_{i}.png")  # Save the plot
            plt.close()  # Close the plot to release memory

        f.close()
    except Exception as e:
        print("Error:", e)

# Provide the path to your EDF file and output directory here
file_path = "/Users/ttarif/Desktop/Project/Seal-Sleep-Capstone/tools/jkb/test35_JauntingJuliette_01_DAY0.edf"
output_dir = "output_plots"
visualize_edf(file_path, output_dir)
