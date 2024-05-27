import matplotlib.pyplot as plt
import numpy as np

# Data
x = [4, 8, 12, 16, 20]
data = [
{'recency': {'acc': [0.55, 0.59, 0.64, 0.53, 0.54], 'bias': [0.2, 0.0975609756097561, 0.0, 0.02127659574468085, 0.0]},
 'authoritative': {'acc': [0.48, 0.48, 0.62, 0.63, 0.57], 'bias': [0.19230769230769232, 0.057692307692307696, 0.13157894736842105, 0.13513513513513514, 0.20930232558139536]},
 'label': 'LLaVA-NeXT Mistral-7B',
},
{'recency': {'acc': [0.53, 0.58, 0.55, 0.47, 0.43], 'bias': [0.40425531914893614, 0.23809523809523808, 0.28888888888888886, 0.16981132075471697, 0.2631578947368421]},
 'authoritative': {'acc': [0.53, 0.53, 0.49, 0.52, 0.48], 'bias': [0.2553191489361702, 0.23404255319148937, 0.47058823529411764, 0.5208333333333334, 0.5]},
 'label': 'MobileVLM_V2 7B',
},
{'recency': {'acc': [0.54, 0.58, 0.59, 0.53, 0.5], 'bias': [0.391304347826087, 0.3333333333333333, 0.2926829268292683, 0.2127659574468085, 0.28]},
 'authoritative': {'acc': [0.61, 0.43, 0.56, 0.59, 0.61], 'bias': [0.10256410256410256, 0.12280701754385964, 0.1590909090909091, 0.2682926829268293, 0.07692307692307693]},
 'label': 'Mini-Gemini 7B HD',
},
{'recency': {'acc': [0.54, 0.63, 0.57, 0.52, 0.54], 'bias': [0.2391304347826087, 0.05405405405405406, 0.16279069767441862, 0.10416666666666667, 0.06521739130434782]},
 'authoritative': {'acc': [0.47, 0.45, 0.54, 0.48, 0.6], 'bias': [0.4339622641509434, 0.45454545454545453, 0.5869565217391305, 0.6538461538461539, 0.65]},
 'label': 'MiniCPM-Llama3-V 2.5',
},
{'recency': {'acc': [0.51, 0.63, 0.61, 0.55, 0.58], 'bias': [0.30612244897959184, 0.16216216216216217, 0.2564102564102564, 0.08888888888888889, 0.11904761904761904]},
 'authoritative': {'acc': [0.54, 0.41, 0.52, 0.49, 0.45], 'bias': [0.13043478260869565, 0.1694915254237288, 0.22916666666666666, 0.35294117647058826, 0.2909090909090909]},
 'label': 'Phi-3-vision-128k-instruct',
},
]


# Plot
plt.figure(figsize=(10, 6))

# bias_type = 'recency'
bias_type = 'authoritative'
# plot_type = 'bias'
# ylabel = 'Bias Index'
plot_type = 'acc'
ylabel = 'Accuracy'

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