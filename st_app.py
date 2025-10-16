import os
import warnings
warnings.filterwarnings("ignore")

# --- Core ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Stats/TS ---
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# --- Data ---
from pymongo import MongoClient
from dotenv import load_dotenv
import pydeck as pdk

# =============================
# 1) Page & Theme
# =============================
st.set_page_config(
    page_title="Traffic Analytics — By Category", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚦 Traffic Analytics")
st.caption("Focus on the selected month with zoomable visuals. Supports STL seasonal/trend decomposition and correlation analysis.")

# =============================
# 2) DB Connection (cached)
# =============================
@st.cache_resource
def get_mongo_client():
    load_dotenv()
    uri = os.getenv("MONGO_URI")
    if not uri:
        st.error("MONGO_URI not found in .env file!")
        return None
    try:
        client = MongoClient(uri)
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

# =============================
# 3) Data Load & Prep (cached)
# =============================
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'traffic_volume (vehicles/hour)': 'traffic_volume',
        'average_speed (km/h)': 'average_speed',
        'Date_Time': 'datetime',
        'Date_time': 'datetime',
        'date_time': 'datetime'
    }
    df = df.rename(columns=rename_map)
    num_cols = ['traffic_volume', 'average_speed', 'incidents', 'latitude', 'longitude']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    if 'incidents' not in df.columns:
        df['incidents'] = 0
    if {'latitude', 'longitude'}.issubset(df.columns):
        df = df.dropna(subset=['latitude', 'longitude'])
    return df

@st.cache_data(ttl=3600)
def load_data(_client, database_name: str, month: str) -> pd.DataFrame:
    if _client is None:
        return pd.DataFrame()
    try:
        db = _client.get_database(database_name)
        if month not in db.list_collection_names():
            return pd.DataFrame()
        collection = db.get_collection(month)
        data = list(collection.find({}))
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        for col in ['_id', 'traffic_id', 'region_id', 'city']:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
        df = _standardize_columns(df)
        if 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['dow'] = df['datetime'].dt.dayofweek
            df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
        return df.drop_duplicates()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

@st.cache_data
def to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')

# Utils
def winsorize(frame: pd.DataFrame, cols, q=(0.01, 0.99)):
    df = frame.copy()
    for c in cols:
        if c in df.columns:
            lo, hi = df[c].quantile(q)
            df[c] = df[c].clip(lo, hi)
    return df

# =============================
# 4) Sidebar Filters
# =============================
client = get_mongo_client()
if not client:
    st.stop()

st.sidebar.header("🔧 Filters")
db_list = ["historical_newyork", "historical_la", "historical_georgia", "historical_sydney"]
selected_db = st.sidebar.selectbox("Dataset", db_list, index=1)

# Month selection with names
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
selected_month = st.sidebar.selectbox("Month", month_names, index=0)

# Data guards moved into an expander
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Advanced Data Cleaning", expanded=True):
    max_speed = st.number_input("Max speed (km/h)", 20, 200, 160, 5, key="max_speed_input")
    max_volume = st.number_input("Max vehicles/hour", 200, 20000, 10000, 100, key="max_volume_input")
    robust_view = st.sidebar.checkbox("Robust view (winsorize 1–99%)", value=True)

# Load
_df = load_data(client, selected_db, selected_month)
if _df.empty:
    st.warning(f"No data found in **{selected_db} / {selected_month}**. Try another selection.")
    st.stop()

# Apply guards
if 'average_speed' in _df.columns:
    _df.loc[_df['average_speed'] > max_speed, 'average_speed'] = np.nan
if 'traffic_volume' in _df.columns:
    _df.loc[_df['traffic_volume'] > max_volume, 'traffic_volume'] = np.nan
if robust_view:
    _df = winsorize(_df, ['traffic_volume', 'average_speed'])

# Region filter
regions = ["(All)"] + sorted(_df.get('region_name', pd.Series(dtype=str)).dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("Regions", regions, default=["(All)"])
if "(All)" not in selected_regions:
    _df = _df[_df['region_name'].isin(selected_regions)]

# =============================
# KPIs
# =============================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows", f"{len(_df):,}", help="Number of rows matching the current filters.")
k2.metric("Avg speed", f"{_df['average_speed'].mean():.1f} km/h", help="Average speed of the currently filtered data.")
k3.metric("Total volume", f"{int(_df['traffic_volume'].sum()):,}", help="Sum of the 'traffic_volume' column.")
k4.metric("Incidents", f"{int(_df['incidents'].sum()):,}", help="Sum of the 'incidents' column.")
if 'datetime' in _df.columns and _df['datetime'].notna().any():
    coverage_days = int((_df['datetime'].max() - _df['datetime'].min()).days) + 1
    k5.metric("Coverage (days)", f"{coverage_days}", help="Includes start and end dates.")
else:
    k5.metric("Coverage (days)", "—", help="No usable timestamps found in the data.")

st.markdown("---")

# =============================
# 5) Tabs by Category
# =============================
TAB_TIME, TAB_SPATIAL, TAB_TREND, TAB_CORR = st.tabs([
    "⏱️ Time Analysis", "📍 Spatial Analysis", "📈 Trend Analysis", "🔗 Correlation Analysis"
])

# ---------- Time Analysis ----------
with TAB_TIME:
    st.subheader("Focus on Selected Month")
    if 'datetime' not in _df.columns:
        st.info("No datetime available.")
    else:
        month_num = month_names.index(selected_month) + 1
        df_m = _df[_df['datetime'].dt.month == month_num].copy()
        years = sorted(df_m['datetime'].dt.year.dropna().unique())
        if not years:
            st.info("No rows for selected month.")
        else:
            c1, c2, c3 = st.columns([1, 1, 1])
            focus_year = c1.selectbox("Year", years, index=len(years) - 1)
            gran = c2.radio("Granularity", ["Hourly", "Daily"], horizontal=True, index=0)
            marks = c3.slider("Mark top highs/lows", 0, 10, 3)

            df_f = df_m[df_m['datetime'].dt.year == focus_year].copy()
            if df_f.empty:
                st.info(f"No data for {focus_year}-{month_num:02d}.")
            else:
                rule = 'H' if gran == "Hourly" else 'D'
                ts = (
                    df_f.set_index('datetime').sort_index()
                    .resample(rule).agg({'traffic_volume': 'mean', 'average_speed': 'mean'})
                )

                def _win(n): return int(np.clip(max(3, n // 20), 3, 24))
                w = _win(len(ts))
                ts['vol_ma'] = ts['traffic_volume'].rolling(w, min_periods=1).mean()
                ts['spd_ma'] = ts['average_speed'].rolling(w, min_periods=1).mean()

                def _mark(s, n):
                    if n <= 0 or s.dropna().empty:
                        return pd.Series(dtype=float), pd.Series(dtype=float)
                    return s.nlargest(n), s.nsmallest(n)

                vol_hi, vol_lo = _mark(ts['traffic_volume'], marks)
                spd_hi, spd_lo = _mark(ts['average_speed'], marks)

                t1, t2 = st.columns(2)
                start, end = ts.index.min(), ts.index.max()

                with t1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts.index, y=ts['traffic_volume'], name='Volume', line=dict(width=1.5)))
                    fig.add_trace(go.Scatter(x=ts.index, y=ts['vol_ma'], name=f'{w}-pt MA', line=dict(width=3)))
                    if marks > 0 and len(vol_hi):
                        fig.add_trace(go.Scatter(x=vol_hi.index, y=vol_hi.values, mode='markers+text', name='Highs',
                                                text=[f"{v:.0f}" for v in vol_hi.values], textposition='middle right',
                                                marker=dict(size=9, symbol='triangle-up')))
                    if marks > 0 and len(vol_lo):
                        fig.add_trace(go.Scatter(x=vol_lo.index, y=vol_lo.values, mode='markers+text', name='Lows',
                                                text=[f"{v:.0f}" for v in vol_lo.values], textposition='middle right',
                                                marker=dict(size=9, symbol='triangle-down')))
                    fig.update_layout(title=f"Traffic Volume — {focus_year}-{month_num:02d} ({gran})", height=460,
                                      xaxis=dict(rangeslider=dict(visible=True), range=[start, end]), yaxis_title='veh/h',
                                      legend=dict(orientation='h', y=1.05))
                    st.plotly_chart(fig, use_container_width=True)

                with t2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts.index, y=ts['average_speed'], name='Avg Speed', line=dict(width=1.5)))
                    fig.add_trace(go.Scatter(x=ts.index, y=ts['spd_ma'], name=f'{w}-pt MA', line=dict(width=3)))
                    if marks > 0 and len(spd_hi):
                        fig.add_trace(go.Scatter(x=spd_hi.index, y=spd_hi.values, mode='markers+text', name='Highs',
                                                text=[f"{v:.1f}" for v in spd_hi.values], textposition='middle left',
                                                marker=dict(size=9, symbol='triangle-up')))
                    if marks > 0 and len(spd_lo):
                        fig.add_trace(go.Scatter(x=spd_lo.index, y=spd_lo.values, mode='markers+text', name='Lows',
                                                text=[f"{v:.1f}" for v in spd_lo.values], textposition='bottom center',
                                                marker=dict(size=9, symbol='triangle-down')))
                    fig.update_layout(title=f"Average Speed — {focus_year}-{month_num:02d} ({gran})", height=460,
                                      xaxis=dict(rangeslider=dict(visible=True), range=[start, end]), yaxis_title='km/h',
                                      legend=dict(orientation='h', y=1.05))
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Seasonal/Trend Decomposition")
    if 'datetime' in _df.columns:
        month_num = month_names.index(selected_month) + 1
        df_m = _df[_df['datetime'].dt.month == month_num].copy()
        years = sorted(df_m['datetime'].dt.year.dropna().unique().tolist())
        if years:
            col_y, col_alg = st.columns([1, 2])
            decomp_year = col_y.selectbox("Year", years, index=len(years) - 1, key="dec_year")
            algo = col_alg.selectbox(
                "Method",
                ["STL (robust)", "Seasonal Decompose (additive)"], index=0,
                help=("STL: Uses LOESS smoothing, less sensitive to outliers. "
                      "Seasonal Decompose: Assumes a fixed seasonal shape, faster.")
            )

            df_my = df_m[df_m['datetime'].dt.year == decomp_year].copy()
            ts_hourly = df_my.set_index('datetime').sort_index()['traffic_volume'].resample('H').mean().ffill()
            n = len(ts_hourly)
            if n >= 48:
                period = 24
                if algo.startswith("STL"):
                    def _odd(k):
                        k = int(max(3, k)); return k if k % 2 == 1 else k + 1
                    seasonal_w = _odd(min(max(11, period), max(7, n // 8)))
                    trend_w    = _odd(min(max(35, period * 5), max(7, n // 2)))
                    st.caption(f"STL params → period={period}, seasonal={seasonal_w}, trend={trend_w}, robust=True")
                    stl = STL(ts_hourly, period=period, seasonal=seasonal_w, trend=trend_w, robust=True)
                    res = stl.fit()
                    obs, trend, seas, resid = ts_hourly, res.trend, res.seasonal, res.resid
                else:
                    dec = seasonal_decompose(ts_hourly, model='additive', period=period)
                    obs, trend, seas, resid = dec.observed, dec.trend, dec.seasonal, dec.resid

                fig = make_subplots(
                    rows=4, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.4, 0.2, 0.2, 0.2],
                    subplot_titles=("Observed", "Trend", "Seasonal", "Residuals")
                )
                fig.add_trace(go.Scatter(x=obs.index, y=obs, name='Observed'), row=1, col=1)
                fig.add_trace(go.Scatter(x=trend.index, y=trend, name='Trend'), row=2, col=1)
                fig.add_trace(go.Scatter(x=seas.index, y=seas, name='Seasonal'), row=3, col=1)
                fig.add_trace(go.Scatter(x=resid.index, y=resid, name='Residuals'), row=4, col=1)
                
                fig.update_layout(height=800, title_text=f"Decomposition using {algo}", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough hourly points in this month/year (need ≥ 48).")
        else:
            st.info("No rows for the selected month.")

# ---------- Spatial Analysis ----------
with TAB_SPATIAL:
    st.subheader("Map Hotspots + Regional Summary")
    
    if {'latitude', 'longitude', 'traffic_volume'}.issubset(_df.columns):
        map_df = _df[['longitude', 'latitude', 'traffic_volume']].dropna()
        if map_df.empty:
            st.warning("No valid geographical data points to display on the map.")
        else:
            map_type = st.radio(
                "Select Map Type",
                ["Heatmap", "Scatterplot", "Hexagon"],
                index=0,
                horizontal=True,
                help=(
                    "**Heatmap**: Shows traffic density. Good for general hotspots.\n\n"
                    "**Scatterplot**: Shows individual data points. Good for detailed analysis.\n\n"
                    "**Hexagon**: Aggregates points into 3D hexagonal bins."
                )
            )

            c1, c2 = st.columns([2, 1])
            with c1:
                tooltip = None
                if map_type == "Heatmap":
                    layer = pdk.Layer(
                        'HeatmapLayer', data=map_df, get_position='[longitude, latitude]',
                        get_weight='traffic_volume', opacity=0.8
                    )
                elif map_type == "Scatterplot":
                    layer = pdk.Layer(
                        'ScatterplotLayer', data=map_df, get_position='[longitude, latitude]',
                        get_fill_color='[255, (1 - traffic_volume / 10000) * 255, 0, 140]',
                        get_radius='(traffic_volume / 80) + 50', pickable=True,
                        radius_min_pixels=3, radius_max_pixels=100
                    )
                    tooltip = {"html": "<b>Traffic Volume:</b> {traffic_volume}"}
                else: # Hexagon
                    layer = pdk.Layer(
                        'HexagonLayer', data=map_df, get_position='[longitude, latitude]',
                        radius=1000, elevation_scale=50, pickable=True, extruded=True,
                    )
                    tooltip = {"html": "<b>Aggregated Volume in Hex:</b> {elevationValue}"}


                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude=map_df['latitude'].mean(),
                        longitude=map_df['longitude'].mean(),
                        zoom=9,
                        pitch=45 if map_type != "Heatmap" else 0,
                    ),
                    layers=[layer],
                    tooltip=tooltip
                ))
            with c2:
                st.info("""
                **Map Interaction**
                - **Zoom**: Scroll in/out.
                - **Pan**: Click and drag.
                - **Rotate**: Ctrl + Click and drag.
                """)
                if 'region_name' in _df.columns:
                    st.metric("Number of Regions Covered", _df['region_name'].nunique())
                st.metric("Geographical Points", f"{len(map_df):,}")

    else:
        st.info("Missing 'latitude', 'longitude', or 'traffic_volume' columns, cannot draw map.")

    st.markdown("---")

    if 'region_name' in _df.columns:
        st.subheader("Volume and Speed Summary by Region")
        agg = _df.groupby('region_name').agg(
            total_volume=('traffic_volume', 'sum'),
            avg_speed=('average_speed', 'mean'),
            incident_count=('incidents', 'sum'),
            record_count=('traffic_volume', 'count')
        ).reset_index().sort_values('total_volume', ascending=False)
        
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(agg, x='total_volume', y='region_name', orientation='h', title='Total Volume by Region')
            fig.update_layout(height=450, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(agg, names='region_name', values='total_volume', title='Volume Share by Region', hole=0.35)
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(agg)
        st.download_button("Download regional summary CSV", data=to_csv_bytes(agg),
                            file_name=f"{selected_db}_{selected_month}_regional_summary.csv")
    else:
        st.info("No 'region_name' column to aggregate by.")

# ---------- Trend Analysis ----------
with TAB_TREND:
    st.subheader("📈 Daily Trends & Weekly Pattern Analysis")

    selected_metric = st.radio(
        "Select Metric",
        ['traffic_volume', 'average_speed'],
        format_func=lambda x: 'Volume' if x == 'traffic_volume' else 'Speed',
        horizontal=True
    )
    
    metric_label = "Median Volume" if selected_metric == 'traffic_volume' else "Median Speed"
    yaxis_title = "veh/h" if selected_metric == 'traffic_volume' else "km/h"

    st.markdown("---")

    st.subheader(f"Weekday vs. Weekend {metric_label} Variation")
    
    if {'dow', 'hour', selected_metric}.issubset(_df.columns):
        df_trend = _df.copy()
        df_trend['day_type'] = df_trend['dow'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

        daily_pattern = df_trend.groupby(['day_type', 'hour'])[selected_metric].median().reset_index()

        fig_line = px.line(
            daily_pattern, x='hour', y=selected_metric, color='day_type',
            markers=True, labels={'hour': 'Hour', selected_metric: yaxis_title, 'day_type': 'Day Type'},
            template='plotly_white'
        )
        
        fig_line.add_vline(x=8, line_width=2, line_dash="dash", line_color="grey", annotation_text="AM Peak")
        fig_line.add_vline(x=17, line_width=2, line_dash="dash", line_color="grey", annotation_text="PM Peak")
        
        fig_line.update_layout(
            height=500, title_text='Daily Trend Comparison: Weekday vs. Weekend',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Missing 'dow', 'hour', or selected metric columns for trend chart.")

    st.markdown("---")
    
    st.subheader("Hourly Median Speed Heatmap")

    if {'dow', 'hour', 'average_speed'}.issubset(_df.columns):
        pivot = _df.pivot_table(index='dow', columns='hour', values='average_speed', aggfunc='median')
        pivot.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][:len(pivot.index)]
        
        fig_heatmap = px.imshow(
            pivot, text_auto=True, aspect='auto',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            labels=dict(color="Median Speed (km/h)", x="Hour", y="Day of Week"),
            template='plotly_white'
        )
        
        fig_heatmap.update_traces(textfont_size=10)
        fig_heatmap.update_layout(height=520)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Missing columns for heatmap.")

# ---------- Correlation Analysis ----------
with TAB_CORR:
    st.subheader("Correlation between Metrics")

    c1, c2, c3 = st.columns([1, 1, 2])
    x_axis = c1.selectbox("X-Axis", ['traffic_volume', 'average_speed', 'incidents'], index=0)
    y_axis = c2.selectbox("Y-Axis", ['traffic_volume', 'average_speed', 'incidents'], index=1)
    
    if x_axis and y_axis:
        if x_axis == y_axis:
            st.warning("Please select two different metrics to compare.")
        else:
            df_sample = _df.sample(min(len(_df), 5000))
            
            fig = px.scatter(
                df_sample, x=x_axis, y=y_axis, trendline="ols",
                title=f"Correlation: {x_axis.replace('_', ' ').title()} vs. {y_axis.replace('_', ' ').title()}"
            )
            st.plotly_chart(fig, use_container_width=True)

            corr = _df[[x_axis, y_axis]].corr().iloc[0, 1]
            st.metric(label=f"Pearson Correlation Coefficient", value=f"{corr:.3f}")
            st.info("The OLS trendline shows the general linear relationship between the two variables.")
    else:
        st.info("Please select both X and Y axes.")