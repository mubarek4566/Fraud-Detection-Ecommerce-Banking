import shap
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def generate_shap_summary1(model, X_train, top_n=20, sample_size=500):
    print("Generating SHAP summary plot...")
    
    # Ensure DataFrame and numeric
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    X_train = X_train.astype(float)
    
    # Subsample for performance
    if X_train.shape[0] > sample_size:
        X_train_sample = X_train.sample(n=sample_size, random_state=42)
    else:
        X_train_sample = X_train

    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_sample)

    # For binary classifiers, get class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Compute top N features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:]
    top_features = X_train_sample.columns[top_indices]

    # Filter for top features
    X_top = X_train_sample[top_features]
    shap_top = shap_values[:, top_indices]

    # Summary plot (beeswarm)
    shap.summary_plot(shap_top, X_top, show=True)


def generate_shap_force_plot(model, X_train, instance_index=0):
    """
    Generate SHAP force plot for a single prediction (local explanation).
    """
    print(f"Generating SHAP force plot for instance {instance_index}...")

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Force plot
    force_plot = shap.plots.force(
        shap_values[instance_index], matplotlib=True, show=False
    )
    
    # Save force plot as image (requires extra steps for interactive HTML)
    force_plot_path = f"outputs/shap_force_plot_instance_{instance_index}.png"
    plt.savefig(force_plot_path)
    plt.close()
    print(f"SHAP force plot saved as '{force_plot_path}'")
