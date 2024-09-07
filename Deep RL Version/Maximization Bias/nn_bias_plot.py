# Final with 400 episodes colored
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_data(files):
    """Load data from multiple files."""
    return [np.load(file) for file in files]

def compute_mean(arrays):
    """Compute mean for each array."""
    return [np.mean(arr, axis=1) for arr in arrays]

def create_dataframe(x, means, labels):
    """Prepare data for plotting."""
    return pd.DataFrame({
        'Episode': np.tile(x, len(labels)),
        'Mean Probability': np.concatenate(means),
        'Method': [label for label, mean in zip(labels, means) for _ in mean]
    })

def plot_data(data):
    """Plot the data using Seaborn with a white background."""
    sns.set(style="white")  # Set background style to plain white
    # Define black and white color palette
    #colors = ['black', 'gray', 'darkgray', 'lightgray']
    #line_styles = ['-', '--', '-.', ':']

    #sns.set(style="whitegrid")  # Set background style to white grid
    plot = sns.lineplot(data=data, x='Episode', y='Mean Probability', hue='Method', style='Method',
                        dashes={'DQN': (2, 2), 'SORDQN': (3, 1), 'DDQN': (5, 2), 'DSORDQN': (1, 1)},
                        linewidth=2,  # Line thickness
                        markers=False)  # No markers

    # Adjust the y-axis limits to zoom into the range where the differences are less noticeable
    y_min, y_max = data['Mean Probability'].min(), data['Mean Probability'].max()
    buffer = (y_max - y_min) * 0.05  # Buffer to slightly expand the y-axis limits
    plt.ylim(y_min - buffer, y_max + buffer)

    # Customize plot appearance
    plt.xlabel('Number of Episodes', fontsize=14, family='Times New Roman')
    plt.ylabel('Probability of a left action', fontsize=14, family='Times New Roman')
    plt.legend(title=None, fontsize='12', loc='best', fancybox=True, shadow=True)  # No legend title
    plt.tick_params(labelsize=12)
    plt.xticks(fontsize=12, family='Times New Roman')
    plt.yticks(fontsize=12, family='Times New Roman')

def main():
    files = [
        "./ProbLeft-Q",
        "./ProbLeft-SORQ",
        "./ProbLeft-D-Q-average",
        "./ProbLeft-SORDQ-average"
    ]

    labels = ['DQN', 'SORDQN', 'DDQN', 'DSORDQN']

    x = np.arange(400)
    arrays = load_data(files)
    means = compute_mean(arrays)

    data = create_dataframe(x, means, labels)

    figsize = (8, 4)
    plt.figure(figsize=figsize)

    plot_data(data)

    plt.tight_layout()
    plt.savefig('Final2.eps', dpi=600, bbox_inches='tight')
    plt.savefig('Final2.pdf', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()