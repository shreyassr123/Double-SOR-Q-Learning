import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import E8

# Example data (replace with your actual data)
mean3 = E8.MeanQ[0][0]
mean5 = E8.MeanQ[1][0]
mean6 = E8.MeanQ[2][0]
mean7 = E8.MeanQ[3][0]

# Set Seaborn style and context
sns.set(style='white')  # Set style to whitegrid for subtle grid lines
sns.set_context('paper', font_scale=1.2)  # Adjust context for paper, increase font size

figsize = 8, 4
fig, ax = plt.subplots(figsize=figsize)

x = [1000*(i+1) for i in range(100)]

# Plotting the lines with increased linewidth

l3, = plt.plot(x, mean3, linewidth=2, color='b', linestyle='-', label='QL')  # Solid blue line
l5, = plt.plot(x, mean5, linewidth=2, color='g', linestyle='--', label='DQL')  # Dashed green line
#l6, = plt.plot(x, mean6, linewidth=2, color='orange', linestyle='-.', label=r"SORQL $w=\frac{1}{1-\gamma}$")  # Dash-dot orange line
#l7, = plt.plot(x, mean7, linewidth=2, color='r', linestyle=':', label=r"DSORQL $w=\frac{1}{1-\gamma}$")  # Dotted red line
l6, = plt.plot(x, mean6, linewidth=2, color='orange', linestyle='-.', label='SORQL (w=10)')
l7, = plt.plot(x, mean7, linewidth=2, color='r', linestyle=':', label= 'SORDQL (w=10)')
# Adding a horizontal line
plt.axhline(y=-0.053, color='black', linestyle='dotted')

# Adjusting legend properties
legend = plt.legend(loc='right', bbox_to_anchor=(0.99, 0.7), fontsize=12)
legend.get_title().set_fontweight('bold')

# Adjusting tick label font size and style
plt.tick_params(labelsize=12)

# Adjusting axis labels to bold
plt.xlabel('Number of Trials', fontsize=14)
plt.ylabel('Expected Profit', fontsize=14)

# Saving the plot as EPS format with higher DPI
plt.tight_layout()
plt.savefig('Fig3.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.savefig('Fig3.eps', format='eps', dpi=600, bbox_inches='tight')
plt.show()
