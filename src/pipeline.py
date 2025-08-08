import pandas as pd
from datetime import datetime
from meteostat import Hourly, Stations
import os

def get_hourly_weather(airports_df: pd.DataFrame, start_date: datetime, end_date: datetime):
    """
    Fetches and caches hourly weather data for the airports in the provided dataframe.
    """
    weather_cache_path = 'data/hourly_weather_summer_2024.csv'
    if os.path.exists(weather_cache_path):
        print("Loading weather data from local cache...")
        weather_df = pd.read_csv(weather_cache_path)
        weather_df['DateTime_UTC'] = pd.to_datetime(weather_df['DateTime_UTC'])
        return weather_df

    print("Fetching weather data from Meteostat (this may take a while)...")
    all_weather_data = []
    
    stations = Stations()
    stations = stations.region('US')

    for i, row in airports_df.iterrows():
        nearby_stations = stations.nearby(row['LATITUDE'], row['LONGITUDE'])
        station = nearby_stations.fetch(1)
        
        if station.empty:
            print(f"({i+1}/{len(airports_df)}) No weather station found for {row['IATA_CODE']}")
            continue
        
        station_id = station.index[0]
        data = Hourly(station_id, start_date, end_date)
        data = data.fetch()
        
        if not data.empty:
            data['Origin'] = row['IATA_CODE']
            all_weather_data.append(data)
        
        print(f"({i+1}/{len(airports_df)}) Fetched weather for {row['IATA_CODE']}")

    weather_df = pd.concat(all_weather_data)
    weather_df = weather_df.reset_index().rename(columns={'time': 'DateTime_UTC'})
    weather_df['DateTime_UTC'] = weather_df['DateTime_UTC'].dt.tz_localize(None)
    weather_df.to_csv(weather_cache_path, index=False)
    print(f"Weather data saved to {weather_cache_path}")
    
    return weather_df

def load_and_prepare_data():
    """
    Loads and processes all data, using a cache for the final processed file.
    """
    processed_data_path = 'data/processed_flight_data.parquet'
    
    # Check if the final, processed data already exists.
    if os.path.exists(processed_data_path):
        print(f"Loading final processed data from {processed_data_path}...")
        return pd.read_parquet(processed_data_path)

    # If not, run the full pipeline...
    print("--- Starting Full Data Pipeline (first run) ---")
    
    print("Loading and combining monthly flight data...")
    flight_files = ['data/flights_2024_06.csv', 'data/flights_2024_07.csv', 'data/flights_2024_08.csv']
    cols_to_use = [
        'FlightDate', 'Reporting_Airline', 'Tail_Number', 'Origin', 'Dest',
        'CRSDepTime', 'DepTime', 'DepDelay', 'ArrDelay', 'AirTime', 'Distance',
        'Cancelled', 'Diverted'
    ]
    flights_df = pd.concat([pd.read_csv(f, usecols=cols_to_use, low_memory=False) for f in flight_files])
    
    flights_df = flights_df[(flights_df['Cancelled'] == 0) & (flights_df['Diverted'] == 0)]
    flights_df = flights_df.drop(columns=['Cancelled', 'Diverted']).dropna()

    print("Identifying top 75 airports...")
    top_airports_list = pd.concat([flights_df['Origin'], flights_df['Dest']]).value_counts().nlargest(75).index.tolist()
    flights_df = flights_df[flights_df['Origin'].isin(top_airports_list) & flights_df['Dest'].isin(top_airports_list)]

    airports_geo_df = pd.read_csv('data/airports_geolocation.csv')
    top_airports_geo = airports_geo_df[airports_geo_df['IATA_CODE'].isin(top_airports_list)]
    
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2024, 8, 31, 23, 59)
    weather_df = get_hourly_weather(top_airports_geo, start_date, end_date)
    
    print("Processing flight timestamps...")
    flights_df['DepTime'] = flights_df['DepTime'].astype(int)
    flights_df['DepTime_str'] = flights_df['DepTime'].apply(lambda x: str(x).zfill(4) if x != 2400 else '0000')
    datetime_str = flights_df['FlightDate'] + ' ' + flights_df['DepTime_str']
    full_datetime = pd.to_datetime(datetime_str, format='%Y-%m-%d %H%M', errors='coerce')
    failed_parsing_mask = full_datetime.isna()
    if failed_parsing_mask.any():
        datetime_str_failed = flights_df.loc[failed_parsing_mask, 'FlightDate'] + ' 0000'
        full_datetime.loc[failed_parsing_mask] = pd.to_datetime(datetime_str_failed, format='%Y-%m-%d %H%M') + pd.Timedelta(days=1)
    flights_df['DateTime_UTC'] = full_datetime.dt.floor('h')
    
    print("Merging flight and weather data...")
    weather_cols_to_merge = {'temp': 'w_temp', 'dwpt': 'w_dwpt', 'rhum': 'w_rhum', 'prcp': 'w_prcp', 'snow': 'w_snow', 'wdir': 'w_wdir', 'wspd': 'w_wspd', 'pres': 'w_pres', 'coco': 'w_coco'}
    weather_df_to_merge = weather_df[['DateTime_UTC', 'Origin'] + list(weather_cols_to_merge.keys())].copy()
    weather_df_to_merge.rename(columns=weather_cols_to_merge, inplace=True)
    merged_df = pd.merge(flights_df, weather_df_to_merge, on=['DateTime_UTC', 'Origin'], how='left')
    
    final_df = merged_df.dropna(subset=['w_temp']).reset_index(drop=True)
    final_df = final_df.drop(columns=['DepTime_str'])

    print(f"Saving final processed data to {processed_data_path}...")
    final_df.to_parquet(processed_data_path)
    
    print("\n--- Data Pipeline Complete ---")
    return final_df