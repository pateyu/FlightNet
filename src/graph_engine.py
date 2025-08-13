import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from src.pipeline import load_and_prepare_data

def create_snapshot_graph(dataframe: pd.DataFrame, timestamp: pd.Timestamp):
    """
    Creates a single hourly graph snapshot for a given timestamp.
    """
    
    # --- 1. Node Mapping ---
    airport_nodes = sorted(dataframe['Origin'].unique())
    airport_map = {airport: i for i, airport in enumerate(airport_nodes)}
    num_nodes = len(airport_nodes)

    # --- 2. Filter Data for Time Windows ---
    feature_window_start = timestamp - pd.Timedelta(hours=3)
    target_window_start = timestamp + pd.Timedelta(hours=1)
    target_window_end = timestamp + pd.Timedelta(hours=4)

    feature_df = dataframe[(dataframe['DateTime_UTC'] >= feature_window_start) & (dataframe['DateTime_UTC'] < timestamp)]
    target_df = dataframe[(dataframe['DateTime_UTC'] >= target_window_start) & (dataframe['DateTime_UTC'] < target_window_end)]

    # --- 3. Calculate Node Features (x) ---
    avg_delay = feature_df.groupby('Origin')['DepDelay'].mean()
    num_departures = feature_df.groupby('Origin').size()
    
    current_weather_df = dataframe[dataframe['DateTime_UTC'] == timestamp].set_index('Origin')
    weather_features = ['w_temp', 'w_wspd', 'w_prcp', 'w_rhum', 'w_pres']
    
    node_features = np.zeros((num_nodes, 2 + len(weather_features)))

    for airport, i in airport_map.items():
        node_features[i, 0] = avg_delay.get(airport, 0)
        node_features[i, 1] = num_departures.get(airport, 0)
        
        if airport in current_weather_df.index:
            weather_data = current_weather_df.loc[airport, weather_features]
            if isinstance(weather_data, pd.Series):
                weather_values = weather_data.values
            else:
                weather_values = weather_data.iloc[0].values
            node_features[i, 2:] = weather_values
            
    x = torch.tensor(node_features, dtype=torch.float)
    
    # --- 4. Define Graph Edges (edge_index) ---
    edges_df = dataframe[['Origin', 'Dest']].drop_duplicates()
    
    source_nodes = edges_df['Origin'].map(airport_map)
    destination_nodes = edges_df['Dest'].map(airport_map)
    
    valid_edges = source_nodes.notna() & destination_nodes.notna()
    source_nodes = source_nodes[valid_edges]
    destination_nodes = destination_nodes[valid_edges]
    
    edge_index = torch.tensor([source_nodes.values, destination_nodes.values], dtype=torch.long)
    
    # --- 5. Define the Prediction Target (y) ---
    target_avg_delay = target_df.groupby('Origin')['DepDelay'].mean()
    
    y = torch.zeros(num_nodes, dtype=torch.float)
    for airport, i in airport_map.items():
        y[i] = target_avg_delay.get(airport, 0)
        
    # --- 6. Assemble the Final Graph Object ---
    graph = Data(x=x, edge_index=edge_index, y=y)
    
    print("Successfully created a PyG Data object:")
    print(graph)

    return graph

# --- To test this file directly ---
if __name__ == '__main__':
    final_df = load_and_prepare_data()
    
    example_timestamp = pd.to_datetime("2024-07-15 12:00:00")
    
    create_snapshot_graph(final_df, example_timestamp)