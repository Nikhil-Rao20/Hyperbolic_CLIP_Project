"""Generate final model performance comparison chart."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Model data (from experiments)
models = [
    "ResNet50\nBaseline",
    "ResNet50\nCross-Gen",
    "CLIP\nZero-Shot",
    "Hyp\nZero-Shot",
    "CLIP\nLinear Probe",
    "CLIP\nFine-Tune",
    "Hyperbolic\nCLIP",
    "Real-Only\n(Euclidean)",
    "Real-Only\n(Hyperbolic)",
]

accuracy = [0.9974, 0.9951, 0.7647, 0.7582, 0.9765, 1.0000, 0.9974, 0.9882, 0.9843]
f1_score = [0.9983, 0.9815, 0.8657, 0.8625, 0.9843, 1.0000, 0.9983, 0.9923, 0.9897]
auroc = [1.0000, 1.0000, 0.8138, 0.6033, 0.9986, 1.0000, 1.0000, 0.9996, 0.9987]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))

bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x, f1_score, width, label='F1 Score', color='#4CAF50', edgecolor='white')
bars3 = ax.bar(x + width, auroc, width, label='AUROC', color='#FF9800', edgecolor='white')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison — Real vs Synthetic MRI Classification', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim(0.5, 1.05)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars for key metrics
for bar in bars1:
    if bar.get_height() >= 0.95:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7, rotation=90)

plt.tight_layout()

out_path = Path(__file__).parent.parent / "assets" / "results" / "model_performance_comparison.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Chart saved to: {out_path}")
