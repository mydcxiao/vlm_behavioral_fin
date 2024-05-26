import matplotlib.pyplot as plt
import numpy as np

# Data
x = [4, 8, 12, 16, 20]
data = [
{'recency': {'acc': [0.55, 0.59, 0.64, 0.53, 0.54], 'bias': [0.2, 0.0975609756097561, 0.0, 0.02127659574468085, 0.0]},
 'authoritative': {'acc': [], 'bias': []},
 'label': 'LLaVA-NeXT Mistral-7B',
},
{'recency': {'acc': [0.53, 0.58, 0.55, 0.47, 0.43], 'bias': [0.40425531914893614, 0.23809523809523808, 0.28888888888888886, 0.16981132075471697, 0.2631578947368421]},
 'authoritative': {'acc': [], 'bias': []},
 'label': 'MobileVLM_V2 7B',
},
{'recency': {'acc': [0.54, 0.58, 0.59, 0.53, 0.5], 'bias': [0.391304347826087, 0.3333333333333333, 0.2926829268292683, 0.2127659574468085, 0.28]},
 'authoritative': {'acc': [], 'bias': []},
 'label': 'Mini-Gemini 7B HD',
},
{'recency': {'acc': [0.54, 0.63, 0.57, 0.52, 0.54], 'bias': [0.2391304347826087, 0.05405405405405406, 0.16279069767441862, 0.10416666666666667, 0.06521739130434782]},
 'authoritative': {'acc': [], 'bias': []},
 'label': 'MiniCPM-Llama3-V 2.5',
},
{'recency': {'acc': [0.51, 0.63, 0.61, 0.55, 0.58], 'bias': [0.30612244897959184, 0.16216216216216217, 0.2564102564102564, 0.08888888888888889, 0.11904761904761904]},
 'authoritative': {'acc': [], 'bias': []},
 'label': 'Phi-3-vision-128k-instruct',
},
]


# Plot
plt.figure(figsize=(10, 6))

bias_type = 'recency'
plot_type = 'acc'
ylabel = 'Wrong by bias'

# Plotting the lines
for model in data:
    plt.plot(x, model[bias_type][plot_type], label=model['label'], linewidth=2, linestyle='-')

# Customizing the plot
plt.title('Line Chart', fontsize=16)
plt.xlabel('Window Size', fontsize=14)
plt.ylabel(ylabel, fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tick_params(length=3, which='both', axis='both', direction='in')
plt.grid(True)
# plt.tight_layout()


# Display the plot
plt.savefig('plot.png',
            # bbox_inches='tight',
            # dpi=300,
            )