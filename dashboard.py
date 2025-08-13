import streamlit as st

st.set_page_config(layout="wide", page_title="FlightNet", initial_sidebar_state="expanded")

st.title("🛫 FlightNet: GNN-Based Delay Propagation Predictor")
st.write("This dashboard visualizes the airport network and predicts future delays using a Graph Attention Network (GAT).")

# --- Sidebar Controls ---
st.sidebar.header("Controls")
selected_date = st.sidebar.date_input("Select Date", value=pd.to_datetime("2024-07-15"))
selected_hour = st.sidebar.slider("Select Hour (UTC)", 0, 23, 12)
st.sidebar.button("Run Prediction")

# --- Main Panel ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Airport Network Graph")
    # Placeholder for the network graph visualization
    st.markdown("*(Network graph will be displayed here)*")
    # Example: net.save_graph("graph.html")
    # st.components.v1.html(open("graph.html", "r").read(), height=615)

with col2:
    st.subheader("Prediction Explorer")
    # Placeholder for airport selection and prediction display
    selected_airport = st.selectbox("Select Airport to Inspect", ["JFK", "LAX", "ORD"])
    st.metric("Predicted Delay (GAT)", "15.2 min")
    st.metric("Predicted Delay (XGBoost)", "18.5 min")
    
    st.subheader("Airport Features")
    st.markdown("*(Current weather and flight data for the selected airport will appear here)*")