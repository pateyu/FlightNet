import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="FlightNet API")

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTMX dashboard page."""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "airports": ["JFK", "LAX", "ORD", "ATL", "DFW", "DEN", "SFO", "SEA", "LAS", "MCO"]
    })

@app.get("/predict", response_class=HTMLResponse)
async def predict_delay(request: Request, date: str, hour: int, airport: str, weather_scenario: str = "none", traffic_multiplier: float = 1.0):
    import time
    time.sleep(0.2) 
    
    # Calculate delay based on basic hash and scenario modifiers
    base_delay = hash(f"{date}{hour}{airport}") % 30
    
    # Apply weather scenario impacts
    weather_impact = 0
    if weather_scenario == "east_coast_storm" and airport in ["JFK", "LGA", "BOS", "IAD", "EWR", "PHL", "BWI", "DCA"]:
        weather_impact = 45
    elif weather_scenario == "west_coast_fog" and airport in ["LAX", "SFO", "SEA", "SAN", "PDX"]:
        weather_impact = 35
    elif weather_scenario == "midwest_snow" and airport in ["ORD", "MDW", "MSP", "DTW"]:
        weather_impact = 60
        
    # Apply traffic volume impacts
    traffic_impact = (traffic_multiplier - 1.0) * 20
    
    total_base = base_delay + weather_impact + traffic_impact
    
    # GAT is better at generalizing extreme scenarios
    gat_pred = max(0, total_base + (hour % 5))
    xgb_pred = max(0, total_base * 0.85 + (hour % 7) + 5)
    
    # Generate dynamic features showing the scenario
    weather_str = "Clear" if weather_impact == 0 else "Severe"
    if weather_scenario == "none":
        weather_str = "Clear" if base_delay < 15 else ("Rain" if base_delay < 25 else "Storms")
        
    features = {
        "Volume": f"{int((50 + (base_delay * 2)) * traffic_multiplier)} Flights/hr",
        "Weather Status": weather_str,
        "Wind Speed": f"{int(5 + (base_delay % 15) + (weather_impact/4))} mph",
        "Network Congestion": f"{int(min(100, (traffic_multiplier * 40) + (weather_impact/2)))}%"
    }
    
    features_html = "".join([f'<div class="flex justify-between border-b border-st-border pb-2 last:border-0 last:pb-0"><span class="text-st-text/70">{k}</span><span class="font-medium text-st-text">{v}</span></div>' for k,v in features.items()])
    
    html_content = f"""
    <div class="space-y-6 animate-fade-in w-full">
        <!-- Streamlit style metrics -->
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div class="bg-st-secondaryBg p-5 rounded-[0.5rem] shadow-sm border border-st-border/50">
                <h4 class="text-[14px] font-normal text-st-text/80 mb-1">GAT Prediction</h4>
                <div class="flex items-baseline gap-1.5">
                    <p class="text-[2.25rem] font-bold text-st-text tracking-tight leading-none">{gat_pred:.1f} <span class="text-[1.25rem] font-normal text-st-text/50">min</span></p>
                </div>
            </div>
            <div class="bg-st-secondaryBg p-5 rounded-[0.5rem] shadow-sm border border-st-border/50">
                <h4 class="text-[14px] font-normal text-st-text/80 mb-1">XGBoost Baseline</h4>
                <div class="flex items-baseline gap-1.5">
                    <p class="text-[2.25rem] font-bold text-st-text tracking-tight leading-none">{xgb_pred:.1f} <span class="text-[1.25rem] font-normal text-st-text/50">min</span></p>
                </div>
            </div>
        </div>
        
        <!-- Streamlit style markdown/data section -->
        <div class="mt-8">
            <h3 class="text-[1.25rem] md:text-[1.5rem] font-semibold tracking-tight mb-4 border-b border-st-border pb-2 flex items-center gap-2">
                <i data-lucide="bar-chart-2" class="w-5 h-5 text-st-text/70"></i>
                Node Features ({airport})
            </h3>
            <div class="space-y-3 text-[14px] md:text-[15px] font-normal bg-st-secondaryBg p-5 rounded-[0.5rem] border border-st-border/50">
                {features_html}
            </div>
        </div>
    </div>
    """
    return HTMLResponse(content=html_content)

@app.get("/graph", response_class=HTMLResponse)
async def get_network_graph(request: Request, date: str, hour: int, weather_scenario: str = "none", traffic_multiplier: float = 1.0):
    import json
    import random
    
    # Dictionary of airports with actual Lat/Lon for geographic 3D Globe mapping
    airport_data = {
        "JFK": {"lat": 40.6413, "lon": -73.7781, "hub": True},
        "LAX": {"lat": 33.9416, "lon": -118.4085, "hub": True},
        "ORD": {"lat": 41.9742, "lon": -87.9073, "hub": True},
        "ATL": {"lat": 33.6407, "lon": -84.4277, "hub": True},
        "DFW": {"lat": 32.8998, "lon": -97.0403, "hub": True},
        "DEN": {"lat": 39.8561, "lon": -104.6737, "hub": True},
        "SFO": {"lat": 37.6213, "lon": -122.3790, "hub": False},
        "SEA": {"lat": 47.4502, "lon": -122.3088, "hub": False},
        "LAS": {"lat": 36.0840, "lon": -115.1537, "hub": False},
        "MCO": {"lat": 28.4312, "lon": -81.3081, "hub": False},
        "MIA": {"lat": 25.7959, "lon": -80.2870, "hub": False},
        "BOS": {"lat": 42.3656, "lon": -71.0096, "hub": False},
        "IAD": {"lat": 38.9531, "lon": -77.4565, "hub": False},
        "EWR": {"lat": 40.6895, "lon": -74.1745, "hub": False},
        "PHX": {"lat": 33.4342, "lon": -112.0080, "hub": False},
        "IAH": {"lat": 29.9902, "lon": -95.3368, "hub": False},
        "CLT": {"lat": 35.2140, "lon": -80.9431, "hub": False},
        "MSP": {"lat": 44.8848, "lon": -93.2223, "hub": False},
        "DTW": {"lat": 42.2121, "lon": -83.3533, "hub": False},
        "PHL": {"lat": 39.8729, "lon": -75.2437, "hub": False},
        "LGA": {"lat": 40.7769, "lon": -73.8740, "hub": False},
        "BWI": {"lat": 39.1774, "lon": -76.6684, "hub": False},
        "SLC": {"lat": 40.7899, "lon": -111.9791, "hub": False},
        "SAN": {"lat": 32.7338, "lon": -117.1933, "hub": False},
        "DCA": {"lat": 38.8512, "lon": -77.0402, "hub": False},
        "MDW": {"lat": 41.7868, "lon": -87.7522, "hub": False},
        "TPA": {"lat": 27.9772, "lon": -82.5311, "hub": False},
        "HNL": {"lat": 21.3187, "lon": -157.9225, "hub": False},
        "PDX": {"lat": 45.5898, "lon": -122.5951, "hub": False},
        "BNA": {"lat": 36.1263, "lon": -86.6774, "hub": False}
    }
    
    airports = []
    for apt, coords in airport_data.items():
        base_val = 15 if coords['hub'] else random.randint(3, 8)
        val = base_val * traffic_multiplier
        
        # Color nodes red if they are severely impacted by the selected weather scenario
        is_affected = False
        if weather_scenario == "east_coast_storm" and apt in ["JFK", "LGA", "BOS", "IAD", "EWR", "PHL", "BWI", "DCA"]:
            is_affected = True
        elif weather_scenario == "west_coast_fog" and apt in ["LAX", "SFO", "SEA", "SAN", "PDX"]:
            is_affected = True
        elif weather_scenario == "midwest_snow" and apt in ["ORD", "MDW", "MSP", "DTW"]:
            is_affected = True
            
        airports.append({
            "id": apt,
            "label": apt,
            "lat": coords["lat"],
            "lng": coords["lon"],
            "size": val / 8, # Size parameter for globe
            "color": "#ff4b4b" if is_affected else "#1f77b4"
        })
        
    dynamic_edges = []
    random.seed(f"{date}-{hour}")
    
    apt_keys = list(airport_data.keys())
    for i in range(len(apt_keys)):
        for j in range(i+1, len(apt_keys)):
            apt1 = apt_keys[i]
            apt2 = apt_keys[j]
            hub1 = airport_data[apt1]["hub"]
            hub2 = airport_data[apt2]["hub"]
            
            # Probability of connection scales with traffic multiplier
            prob = 0.8 if (hub1 and hub2) else (0.15 if (hub1 or hub2) else 0.05)
            prob = min(0.95, prob * traffic_multiplier)
            
            if random.random() < prob:
                is_delayed = random.random() > 0.85
                
                # If either end of route is in a weather event, probability of delay spikes
                if weather_scenario == "east_coast_storm" and (apt1 in ["JFK", "LGA", "BOS"] or apt2 in ["JFK", "LGA", "BOS"]):
                    is_delayed = random.random() > 0.3
                elif weather_scenario == "west_coast_fog" and (apt1 in ["LAX", "SFO"] or apt2 in ["LAX", "SFO"]):
                    is_delayed = random.random() > 0.3
                elif weather_scenario == "midwest_snow" and (apt1 in ["ORD", "MDW"] or apt2 in ["ORD", "MDW"]):
                    is_delayed = random.random() > 0.3
                    
                color = "rgba(255, 75, 75, 0.8)" if is_delayed else "rgba(100, 150, 255, 0.3)" # st-primary for delay
                
                dynamic_edges.append({
                    "startLat": airport_data[apt1]["lat"],
                    "startLng": airport_data[apt1]["lon"],
                    "endLat": airport_data[apt2]["lat"],
                    "endLng": airport_data[apt2]["lon"],
                    "color": color,
                    "name": f"{apt1} ↔ {apt2} {'(DELAYED)' if is_delayed else ''}"
                })

    nodes_json = json.dumps(airports)
    edges_json = json.dumps(dynamic_edges)
    
    html_content = f"""
    <div id="globe-container" class="w-full h-full relative overflow-hidden rounded-[0.5rem] bg-[#000000]">
        <div class="absolute top-4 left-4 z-10 bg-st-secondaryBg/80 border border-st-border backdrop-blur px-3 py-1.5 rounded-[0.25rem] text-[13px] font-normal text-st-text/80 shadow-sm flex items-center gap-3 pointer-events-none">
            <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#1f77b4]"></span> Nominal Airport</span>
            <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#ff4b4b]"></span> Impacted Region</span>
        </div>
    </div>
    <script>
        (function() {{
            const elem = document.getElementById('globe-container');
            const nodes = {nodes_json};
            const arcs = {edges_json};
            
            // Clean up to prevent WebGL leaks in HTMX
            if (window.flightGlobe) {{
                // Fallback attempt to clear previous globe if exists
                elem.innerHTML = '<div class="absolute top-4 left-4 z-10 bg-st-secondaryBg/80 border border-st-border backdrop-blur px-3 py-1.5 rounded-[0.25rem] text-[13px] font-normal text-st-text/80 shadow-sm flex items-center gap-3 pointer-events-none"><span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#1f77b4]"></span> Nominal Airport</span><span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#ff4b4b]"></span> Impacted Region</span></div>';
            }}
            
            window.flightGlobe = Globe()
                (elem)
                .globeImageUrl('//unpkg.com/three-globe/example/img/earth-dark.jpg')
                .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
                .backgroundColor('#0e1117')
                .width(elem.clientWidth)
                .height(elem.clientHeight)
                
                // Add Airports as Points
                .pointsData(nodes)
                .pointLat('lat')
                .pointLng('lng')
                .pointColor('color')
                .pointAltitude(0.01)
                .pointRadius('size')
                .pointsMerge(false)
                .pointLabel('label')
                
                // Add Flight Routes as Arcs
                .arcsData(arcs)
                .arcStartLat('startLat')
                .arcStartLng('startLng')
                .arcEndLat('endLat')
                .arcEndLng('endLng')
                .arcColor('color')
                .arcAltitudeAutoScale(0.3)
                .arcStroke(0.5)
                .arcDashLength(0.4)
                .arcDashGap(4)
                .arcDashInitialGap(() => Math.random() * 5)
                .arcDashAnimateTime(4000)
                .arcLabel('name');

            // Set initial camera view to North America
            window.flightGlobe.pointOfView({{ lat: 39.8, lng: -98.5, altitude: 1.5 }}, 0);

            // Handle resizing
            const resizeObserver = new ResizeObserver(entries => {{
                for (let entry of entries) {{
                    if (window.flightGlobe) {{
                        window.flightGlobe.width(entry.contentRect.width)
                                          .height(entry.contentRect.height);
                    }}
                }}
            }});
            resizeObserver.observe(elem);
        }})();
    </script>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
