import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Load the CSV file
file_path = 'checkpoint/mmwhs/DDSPSeg_20250526_222743_A2B/fold_0/train_DDSPSeg.csv'
df = pd.read_csv(file_path)

# Define attributes to plot and corresponding titles
attributes = ['loss_ent', 'loss_consist', 'loss_fe', 'loss_seg', 'loss_all']
titles = [f"{attr} curve" for attr in attributes]

# Prepare the magma colormap
colors = cm.magma(np.linspace(0, 1, len(attributes)+4))[1:6, :3]  # Select 5 colors from the magma colormap

# Plotting
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Training Loss Curves', fontsize=16)

for i, attr in enumerate(attributes):
    axes[i].plot(df['epoch'], df[attr], color=colors[i])
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel(attr)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
