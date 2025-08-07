import pandas as pd
import torch
from torch_geometric.data import Data

def create_snapshot_graph(dataframe: pd.DataFrame, timestamp: pd.Timestamp):
    """
    Creates a single graph snapshot for a given timestamp.
    
    Args:
        dataframe (pd.DataFrame): The pre-processed dataframe of all flights.
        timestamp (pd.Timestamp): The specific time 'T' for which to create the snapshot.

    Returns:
        A PyG Data object representing the graph at that timestamp.
    """
    
    # --- 1. Node Mapping ---
    airport_nodes = sorted(pd.concat([dataframe['Origin'], dataframe['Dest']]).unique())
    airport_map = {airport: i for i, airport in enumerate(airport_nodes)}
    num_nodes = len(airport_nodes)

    # --- 2. Filter Data for Time Windows ---
    # Define the time windows relative to the snapshot timestamp
    feature_window_start = timestamp - pd.Timedelta(hours=3)
    target_window_end = timestamp + pd.Timedelta(hours=3)

    # Filter the dataframe to get flights within these windows
    feature_df = dataframe[(dataframe['FlightDate'] >= feature_window_start) & (dataframe['FlightDate'] < timestamp)]
    target_df = dataframe[(dataframe['FlightDate'] >= timestamp) & (dataframe['FlightDate'] < target_window_end)]

    print(f"Found {len(feature_df)} flights in the 3-hour feature window.")
    print(f"Found {len(target_df)} flights in the 3-hour target window.")

    # --- TODO: The next steps will go here ---
    # 3. Calculate node features (x)
    # 4. Define edges (edge_index)
    # 5. Define the prediction target (y)

    return None


# --- To test this file directly ---
if __name__ == '__main__':
    from pipeline import load_and_prepare_data
    
    final_df = load_and_prepare_data()
    
    # Let's pick a specific timestamp from our data
    example_timestamp = pd.to_datetime("2023-07-15 12:00:00")
    
    create_snapshot_graph(final_df, example_timestamp)