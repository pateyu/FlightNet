import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.pipeline import load_and_prepare_data
from src.graph_engine import create_snapshot_graph
from src.model import GAT

def train(model, data_list, optimizer, criterion, device):
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
def test(model, data_list, criterion, device):
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

    num_node_features = graph_dataset[0].x.shape[1]
    
    model = GAT(in_channels=num_node_features, hidden_channels=64, out_channels=1, heads=4)
    model.to(device) # --- NEW: Move the model to the selected device ---
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    print("\n--- Starting Model Training ---")
    for epoch in range(1, 101):
        
        train_loss = train(model, train_dataset, optimizer, criterion, device)
        test_loss = test(model, test_dataset, criterion, device)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    print("\n--- Training Complete ---")