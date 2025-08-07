from src.pipeline import load_and_prepare_data

if __name__ == "__main__":
    print("Running the data pipeline...")
    final_df = load_and_prepare_data()
    
    print("\n--- Pipeline Test Successful ---")
    print(f"Loaded {len(final_df)} flights.")
    print("Sample of the final DataFrame:")
    print(final_df.head())