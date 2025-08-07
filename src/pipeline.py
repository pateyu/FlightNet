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
        return pd.read_csv(weather_cache_path)

    print("Fetching weather data from Meteostat (this may take a while)...")
    all_weather_data = []

    for i, row in airports_df.iterrows():
        station = Stations()
        # Find stations near the airport's lat/lon
        station = station.nearby(row['LATITUDE'], row['LONGITUDE'])
        station_id = station.index[0]
        
        # Get hourly data for the date range
        data = Hourly(station_id, start_date, end_date)
        data = data.fetch()
        
        if not data.empty:
            data['Origin'] = row['IATA_CODE'] # Use airport code for joining
            all_weather_data.append(data)
        
        print(f"({i+1}/{len(airports_df)}) Fetched weather for {row['IATA_CODE']}")

    weather_df = pd.concat(all_weather_data)
    # Convert time to timezone-naive UTC for merging
    weather_df = weather_df.reset_index().rename(columns={'time': 'DateTime_UTC'})
    weather_df['DateTime_UTC'] = weather_df['DateTime_UTC'].dt.tz_convert(None)

    weather_df.to_csv(weather_cache_path, index=False)
    print(f"Weather data saved to {weather_cache_path}")
    
    return weather_df

def load_and_prepare_data():
    """
    Loads, combines, and processes all raw data for the project.
    """
    print("--- Starting Data Pipeline ---")
    
    # 1. Load and combine the three monthly flight CSVs
    print("Loading and combining monthly flight data...")
    flight_files = ['data/flights_2024_06.csv', 'data/flights_2024_07.csv', 'data/flights_2024_08.csv']
    # Select only the columns we need to save memory
    cols_to_use = ['FlightDate', 'Reporting_Airline', 'Tail_Number', 'Origin', 'Dest', 'CRSDepTime', 'DepTime', 'DepDelay']
    
    flights_df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols_to_use + ['Cancelled', 'Diverted'], low_memory=False) for f in flight_files])
    
    # 2. Filter out cancelled/diverted flights
    flights_df = flights_df[(flights_df['Cancelled'] == 0) & (flights_df['Diverted'] == 0)]
    flights_df = flights_df[cols_to_use].dropna()

    # 3. Identify Top 75 Airports
    print("Identifying top 75 airports...")
    top_airports_list = pd.concat([flights_df['Origin'], flights_df['Dest']]).value_counts().nlargest(75).index.tolist()
    flights_df = flights_df[flights_df['Origin'].isin(top_airports_list) & flights_df['Dest'].isin(top_airports_list)]

    # 4. Fetch Weather Data
    airports_geo_df = pd.read_csv('data/airports_geolocation.csv')
    top_airports_geo = airports_geo_df[airports_geo_df['IATA_CODE'].isin(top_airports_list)]
    
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2024, 8, 31, 23, 59)
    weather_df = get_hourly_weather(top_airports_geo, start_date, end_date)
    weather_df['DateTime_UTC'] = pd.to_datetime(weather_df['DateTime_UTC'])
    
    # 5. Process Timestamps for Flights
    print("Processing flight timestamps...")
    # Pad time strings to 4 digits (e.g., '630' -> '0630')
    flights_df['DepTime_str'] = flights_df['DepTime'].astype(int).astype(str).str.zfill(4)
    # Combine date and time to create a full timestamp
    full_datetime = pd.to_datetime(flights_df['FlightDate'] + ' ' + flights_df['DepTime_str'], format='%Y-%m-%d %H%M', errors='coerce')
    # Round down to the nearest hour for merging with hourly weather
    flights_df['DateTime_UTC'] = full_datetime.dt.floor('H')
    
    # 6. Merge Flights with Weather
    print("Merging flight and weather data...")
    # Rename weather columns to avoid conflicts
    weather_cols_to_merge = {'temp': 'w_temp', 'dwpt': 'w_dwpt', 'rhum': 'w_rhum', 'prcp': 'w_prcp', 'snow': 'w_snow', 'wdir': 'w_wdir', 'wspd': 'w_wspd', 'pres': 'w_pres', 'coco': 'w_coco'}
    weather_df.rename(columns=weather_cols_to_merge, inplace=True)
    
    final_df = pd.merge(flights_df, weather_df[['DateTime_UTC', 'Origin'] + list(weather_cols_to_merge.values())], on=['DateTime_UTC', 'Origin'], how='left')
    
    final_df = final_df.dropna().reset_index(drop=True)
    
    print("\n--- Data Pipeline Complete ---")
    return final_df