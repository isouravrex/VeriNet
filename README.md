# VeriNet Agent
> **Autofix High Volume Networks**

![VeriNet Logo](https://via.placeholder.com/800x200.png?text=VeriNet+Agent+Logo+Placeholder)
*(Replace the image link above with your actual logo image file when ready)*

---

## 📋 About This Prototype

This repository contains the interactive prototype for **VeriNet Agent**, an autonomous system designed to ensure network fidelity in high-volume infrastructure.

Unlike traditional monitoring tools that passively alert on thresholds, VeriNet acts as an active "immune system." It uses **Agentic AI** and custom **correlation algorithms** to continuously compare live network telemetry against a "Digital Twin" (our Gold Standard). When configuration drift or performance degradation is detected, the agent calculates a **Fidelity Score** and autonomously simulates remediation commands to "heal" the network.

**This Notebook environment is designed to:**
1.  Generate high-volume synthetic network telemetry.
2.  Profile the data to understand underlying correlations.
3.  Run the VeriNet Agent core loop to visualize drift detection and autonomous recovery in real-time.

---

## ⚙️ Prerequisites

To run this prototype, you do not need any local setup. The preferred environment is **Google Colab**.

* A Google Account (to access Google Colab).
* (Optional) A sample CSV file if you wish to test the Data Profiling section with your own data.

---

## 🚀 Quick Start Guide (Google Colab)

The easiest way to test the agent is to run the entire notebook top-to-bottom.

1.  **Open the Notebook:** Click the badge below to open the `VeriNet_Agent_Prototype.ipynb` in Google Colab.
    [![](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
    *(**Note:** Replace `YOUR_COLAB_LINK_HERE` with the actual shareable link to your notebook).*

2.  **Connect to Runtime:** In the top right corner of Colab, click **"Connect"** to allocate server resources.

3.  **Run All Cells:** In the menu bar, go to **Runtime** > **Run all**.

The notebook will begin installing necessary dependencies (like `ydata-profiling`), generating synthetic data, and initiating the agent's monitoring loop. Scroll down to follow the execution.

---

## 📓 Notebook Structure & Walkthrough

Here is what you will find inside the notebook and what to look for during testing:

### Phase 1: Setup & Synthetic Data Generation
We initialize the Python environment and define the parameters for our "Digital Twin." We then generate high-volume synthetic data simulating normal traffic, jitter, and eventually, induced faults (drift).

### Phase 2: Exploratory Data Analysis (Data Profiling)
Before running the agent, we analyze the data structure. If you have a `sample.csv`, upload it to the Colab file area on the left sidebar.

We use `ydata-profiling` to generate an interactive report within the notebook. This helps identify the correlations (e.g., "Does high CPU always correlate with packet loss?") that drive the agent's logic.

```python
# Example code used in the notebook for profiling
import pandas as pd
from ydata_profiling import ProfileReport

try:
    # Ensure sample.csv is uploaded to the Colab session storage first
    df_profile = pd.read_csv('sample.csv')
    profile = ProfileReport(df_profile, title="VeriNet Data Profile", explorative=True)
    profile.to_notebook_iframe()
except FileNotFoundError:
    print("sample.csv not found. Skipping external data profiling.")
