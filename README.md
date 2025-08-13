# 🛫 FlightNet: GNN-Based Flight Delay Propagation Predictor

## Project Goal
The goal of this project is to build a Graph Attention Network (GAT) to model and predict the propagation of flight delays across the U.S. aviation network. By representing airports as nodes and flight routes as edges, the model learns complex spatiotemporal patterns from real-world flight and weather data to forecast future delay events.

## Key Features
- **Graph-Based Modeling:** Represents the U.S. airport system as a dynamic graph, with airports as nodes and flight routes as edges.
- **Real-World Data Fusion:** Integrates granular, hourly flight data from the Bureau of Transportation Statistics (BTS) with corresponding hourly weather data fetched programmatically from Meteostat.
- **Advanced GNN Architecture:** Employs a Graph Attention Network (GAT) to learn how influential each connected airport is in propagating delays.
- **Temporal Snapshotting:** Models the network's evolution by generating and training on hourly graph "snapshots", capturing the state of the network at any given time.

## Tech Stack
- **Python**
- **PyTorch & PyTorch Geometric** for GNN modeling
- **XGBoost** for baseline comparison
- **Pandas** for data manipulation
- **Meteostat** for weather data collection
- **Streamlit** for the interactive dashboard

## Results
The GAT model was trained on a dataset of hourly graph snapshots from Summer 2024. It was benchmarked against a strong XGBoost baseline model that used the same features but lacked the graph's structural information.

| Model | Test MSE (Scaled) |
| :--- | :---: |
| **GAT** | |
| **XGBoost** | |

The results demonstrate that the GAT's ability to model network relationships provides a significant performance improvement over traditional models.

### Dashboard Preview



## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/FlightNet.git](https://github.com/your-username/FlightNet.git)
    cd FlightNet
    ```
2.  **Create and activate the Conda environment:**
    ```bash
    mamba env create -f environment.yml
    mamba activate flightnet
    ```
3.  **Install any additional packages (if needed):**
    This step is a fallback for any packages that had issues installing via the `.yml` file.
    ```bash
    pip install torch_geometric
    ```
4.  **Run the training script:**
    This script will run the full data pipeline (loading from cache if available), train both the GAT and XGBoost models, and save the trained models.
    ```bash
    python train.py
    ```
5.  **Launch the dashboard:**
    ```bash
    streamlit run dashboard.py
    ```