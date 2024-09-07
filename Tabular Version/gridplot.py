import grid as E1
import seaborn as sns
import matplotlib.pyplot as plt

# Set the seaborn style and matplotlib parameters for better aesthetics
sns.set(style="white")
plt.rcParams.update({
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 13,
    'figure.dpi': 600,  # Higher DPI for better quality
    'text.usetex': True,
    'font.family': 'serif'
})

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(8, 4.5)) # Aspect ratio modified to make the figure look good in the double page format

# Set the title and labels

ax.set_xlabel(r'$\mathrm{Number}$ $\mathrm{of}$ $\mathrm{steps}$', fontsize=16)
ax.set_ylabel(r'$\mathrm{Reward}$ $\mathrm{per}$ $\mathrm{step}$', fontsize=17)

# Define specific colors for each line
colors = {
    'QL': 'blue',
    'MF-SORQL': 'orange',
    'DQL': 'green',
    'MF-DSORQL': 'red',
}

# Plot Q and Double Q for the first dataset with the specified colors
sns.lineplot(x=[100 * (i + 1) for i in range(100)], y=[r / 100. for r in E1.R[0]], ax=ax, label='QL', linestyle='-', color=colors['QL'], linewidth=2.0)
sns.lineplot(x=[100 * (i + 1) for i in range(100)], y=[r / 100. for r in E1.R[1]], ax=ax, label='MF-SORQL', linestyle='--', color=colors['MF-SORQL'], linewidth=2.0)
sns.lineplot(x=[100 * (i + 1) for i in range(100)], y=[r / 100. for r in E1.R[2]], ax=ax, label='DQL', linestyle='-.', color=colors['DQL'], linewidth=2.0)
sns.lineplot(x=[100 * (i + 1) for i in range(100)], y=[r / 100. for r in E1.R[3]], ax=ax, label='MF-DSORQL', linestyle=':', color=colors['MF-DSORQL'], linewidth=2.0)

# Customize the x and y limits and ticks
ax.set_xlim(0, 10000)
ax.set_ylim(-1.15, 0.2)

ax.set_yticks([-1, -0.5, 0])
ax.set_yticklabels(['$-1.0$', '$-0.5$', '$0.0$'])



# Add legend with adjusted aesthetics
ax.legend(loc='upper left', frameon=True, shadow=True, borderpad=0.5, labelspacing=0.5, borderaxespad=0.3)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("Grid.pdf", format="pdf")

plt.show()
