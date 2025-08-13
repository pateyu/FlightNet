import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from src.pipeline import load_and_prepare_data
from src.graph_engine import create_snapshot_graph
from src.model import GAT

def train_gat(model, data_list, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in data_list:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.squeeze(), data.y)
        if torch.isnan(loss):
            continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_list)

@torch.no_grad()
def test_gat(model, data_list, criterion, device):
    model.eval()
    total_loss = 0
    for data in data_list:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        loss = criterion(out.squeeze(), data.y)
        if not torch.isnan(loss):
            total_loss += loss.item()
    return total_loss / len(data_list)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    final_df = load_and_prepare_data()

    print("\n--- Generating Graph Snapshots ---")
    timestamps = final_df['DateTime_UTC'].unique()
    sample_timestamps = pd.to_datetime(timestamps)[::12]
    
    graph_dataset = []
    for ts in tqdm(sample_timestamps, desc="Creating graphs"):
        graph = create_snapshot_graph(final_df, ts)
        if graph is not None and graph.x.size(0) > 0:
            graph_dataset.append(graph)
    
    print(f"\nCreated a dataset with {len(graph_dataset)} graph snapshots.")

    print("Scaling node features and targets...")
    all_x = torch.cat([data.x for data in graph_dataset], dim=0).numpy()
    all_y = torch.cat([data.y.unsqueeze(1) for data in graph_dataset], dim=0).numpy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(all_x)
    y_scaler.fit(all_y)

    for data in graph_dataset:
        data.x = torch.from_numpy(x_scaler.transform(data.x)).float()
        data.y = torch.from_numpy(y_scaler.transform(data.y.unsqueeze(1))).float().squeeze()
    
    split_index = int(len(graph_dataset) * 0.8)
    train_dataset = graph_dataset[:split_index]
    test_dataset = graph_dataset[split_index:]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # --- 1. GAT Model Training ---
    print("\n--- Starting GAT Model Training ---")
    num_node_features = graph_dataset[0].x.shape[1]
    gat_model = GAT(in_channels=num_node_features, hidden_channels=64, out_channels=1, heads=4)
    gat_model.to(device)
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, 101):
        train_loss = train_gat(gat_model, train_dataset, optimizer, criterion, device)
        test_loss = test_gat(gat_model, test_dataset, criterion, device)
        if epoch % 10 == 0:
            print(f'GAT Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    final_gat_loss = test_gat(gat_model, test_dataset, criterion, device)
    print(f"\n--- GAT Training Complete ---")
    print(f"Final GAT Test MSE (scaled): {final_gat_loss:.4f}")

    # --- 2. XGBoost Baseline Training ---
    print("\n--- Starting XGBoost Baseline Training ---")
    # Convert graph data to a tabular format for XGBoost
    X_train = np.concatenate([data.x.numpy() for data in train_dataset], axis=0)
    y_train = np.concatenate([data.y.numpy() for data in train_dataset], axis=0)
    X_test = np.concatenate([data.x.numpy() for data in test_dataset], axis=0)
    y_test = np.concatenate([data.y.numpy() for data in test_dataset], axis=0)

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    predictions = xgb_model.predict(X_test)
    xgb_loss = mean_squared_error(y_test, predictions)

    print(f"--- XGBoost Training Complete ---")
    print(f"Final XGBoost Test MSE (scaled): {xgb_loss:.4f}")

    # --- 3. Final Comparison ---
    print("\n--- Model Comparison ---")
    print(f"GAT Test MSE: {final_gat_loss:.4f}")
    print(f"XGBoost Test MSE: {xgb_loss:.4f}")

    if final_gat_loss < xgb_loss:
        print("\nConclusion: The GAT model outperformed the XGBoost baseline. ✅")
    else:
        print("\nConclusion: The XGBoost baseline outperformed the GAT model. 🟡")