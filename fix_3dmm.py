# Removing Wobbles: Script to apply hard smoothing to the z-translations of a 3DMM model for multiple time ranges

import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

def hard_smooth_translations(trans, fps, t1, t2):
    """
    Completely smooths z-motions between time t1 and t2 in place.
    
    Args:
    - trans (np.ndarray): A NumPy array of shape (N, 3) representing translations (x, y, z).
    - fps (int): Frames per second of the video.
    - t1 (float): Start time (in seconds) for smoothing.
    - t2 (float): End time (in seconds) for smoothing.

    Modifies:
    - trans (np.ndarray): Updates the z-component of translations between t1 and t2.
    """
    # Convert t1 and t2 from seconds to frame indices
    frame_t1 = int(t1 * fps)
    frame_t2 = int(t2 * fps)

    # Extract the z-component
    trans_z = trans[:, 2]

    # Smooth the z-component between t1 and t2 by linearly interpolating between the values
    z_t1 = trans_z[frame_t1]
    z_t2 = trans_z[frame_t2]
    trans_z[frame_t1:frame_t2] = np.linspace(z_t1, z_t2, frame_t2 - frame_t1)

    # Apply additional smoothing around the transition points
    start_smooth = max(0, frame_t1 - fps // 2)
    end_smooth = min(frame_t2 + fps // 2, len(trans_z))
    trans_z[start_smooth:end_smooth] = savgol_filter(trans_z[start_smooth:end_smooth], fps, 3)


def apply_smoothing_multiple_ranges(trans, fps, time_ranges):
    """
    Applies hard smoothing for multiple time ranges on the z-translation and updates the original tensor.

    Args:
    - trans (np.ndarray): A NumPy array of shape (N, 3) representing translations (x, y, z).
    - fps (int): Frames per second of the video.
    - time_ranges (list of tuples): List of (t1, t2) time ranges (in seconds) for smoothing.

    Modifies:
    - trans (np.ndarray): The z-component of translations is smoothed in place for each time range.
    """
    for t1, t2 in time_ranges:
        hard_smooth_translations(trans, fps, t1, t2)


def plot_smoothing_results(trans, original_trans, fps, time_ranges, vis_path):
    """
    Plots the original z-translations and smoothed z-translations for each time range.

    Args:
    - trans (np.ndarray): The updated translation array.
    - original_trans (np.ndarray): The original translation array.
    - fps (int): Frames per second of the video.
    - time_ranges (list of tuples): List of (t1, t2) time ranges (in seconds) for smoothing.
    """
    trans_z_old = original_trans[:, 2]
    trans_z_new = trans[:, 2]
    time_steps = np.arange(len(trans_z_old)) / fps

    # Determine the number of subplots based on the number of time ranges
    num_plots = len(time_ranges)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

    # Plot the original and smoothed translations for each time range
    for idx, (t1, t2) in enumerate(time_ranges):
        frame_t1 = int(t1 * fps)
        frame_t2 = int(t2 * fps)
        start_plot = max(0, frame_t1 - fps)
        end_plot = min(len(trans_z_old), frame_t2 + fps)

        if num_plots == 1:
            ax = axs  # When there's only one subplot, axs isn't an array
        else:
            ax = axs[idx]

        time_plot = time_steps[start_plot:end_plot]
        ax.plot(time_plot, trans_z_old[start_plot:end_plot], label='Original Z Translation', color='blue', alpha=0.5)
        ax.plot(time_plot, trans_z_new[start_plot:end_plot], label=f'Smoothed Z (t1={t1}, t2={t2})', color='red', linestyle='--')

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Z Translation')
        ax.legend()
        ax.grid(True)
        ax.set_title(f'Original vs Smoothed Z Translation (t1={t1}, t2={t2})')

    plt.tight_layout()
    plt.savefig(vis_path)
    plt.show()


# Load the data
file_path = "data/processed/videos/Girish2_512/coeff_fit_mp.npy"
updated_file_path = "coeff_fit_mp_updated_multiple_smoothings.npy"
vis_path_file = "Old Vs Updated multiple_smoothing_subplots.png"
coeff_fit_mp = np.load(file_path, allow_pickle=True).tolist()

# Extract the 'trans' array and keep it as NumPy array
trans = coeff_fit_mp['trans']

# Make a copy of the original translations for comparison
original_trans = trans.copy()

# Define parameters
fps = 25
time_ranges = [(3.0, 5.2), (29.0, 30.5)]  # List of time ranges for smoothing

# Apply smoothing for multiple ranges, updating 'trans' in place
apply_smoothing_multiple_ranges(trans, fps, time_ranges)

# Save the updated translation back to the file
np.save(updated_file_path, coeff_fit_mp)

# Plot the results comparing original and smoothed translations
plot_smoothing_results(trans, original_trans, fps, time_ranges, vis_path_file)
