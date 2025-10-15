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
st.title("🚦 Traffic Analytics — By Category (Time / Spatial / Trends / Correlation)")
st.caption("Focus on the selected month with zoomable visuals. Supports STL seasonal/trend decomposition and correlation analysis with external factors.")

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


# Data guards
st.sidebar.markdown("---")
st.sidebar.header("🧹 Data Quality")
max_speed = st.sidebar.number_input("Max speed (km/h)", 20, 200, 160, 5)
max_volume = st.sidebar.number_input("Max vehicles/hour", 200, 20000, 10000, 100)
robust_view = st.sidebar.checkbox("Robust view (winsorize 1–99%)", value=True)

# Load
_df = load_data(client, selected_db, selected_month)
if _df.empty:
    st.warning(f"No data found in **{selected_db} / {selected_month}**")
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
# KPIs（含小問號說明）
# =============================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows", f"{len(_df):,}", help="Number of rows matching the current filters (Dataset / Month / Regions / Data Quality).")
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
TAB_TIME, TAB_SPATIAL, TAB_TREND = st.tabs([
    "⏱️ Time Analysis", "📍 Spatial Analysis", "📈 Trend Analysis"
])

# ---------- 時間分析 ----------
with TAB_TIME:
    st.subheader("Focus on Selected Month (Switch Year/Granularity/Extreme Markers)")
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

                # dynamic smoother window
                def _win(n): return int(np.clip(max(3, n // 20), 3, 24))
                w = _win(len(ts))
                ts['vol_ma'] = ts['traffic_volume'].rolling(w, min_periods=1).mean()
                ts['spd_ma'] = ts['average_speed'].rolling(w, min_periods=1).mean()

                # helper to mark extremes
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
                                                text=[f"{v:.0f}" for v in vol_hi.values], textposition='middle right', # avoid overlap
                                                marker=dict(size=9, symbol='triangle-up')))                            # textposition='bottom center' to textposition='middle right'
                    if marks > 0 and len(vol_lo):
                        fig.add_trace(go.Scatter(x=vol_lo.index, y=vol_lo.values, mode='markers+text', name='Lows',
                                                 text=[f"{v:.0f}" for v in vol_lo.values], textposition='middle right', # avoid overlap
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
                                                 text=[f"{v:.1f}" for v in spd_hi.values], textposition='middle left', # avoid overlap
                                                 marker=dict(size=9, symbol='triangle-up')))
                    if marks > 0 and len(spd_lo):
                        fig.add_trace(go.Scatter(x=spd_lo.index, y=spd_lo.values, mode='markers+text', name='Lows',
                                                 text=[f"{v:.1f}" for v in spd_lo.values], textposition='bottom center', # avoid overlap
                                                 marker=dict(size=9, symbol='triangle-down')))
                    fig.update_layout(title=f"Average Speed — {focus_year}-{month_num:02d} ({gran})", height=460,
                                      xaxis=dict(rangeslider=dict(visible=True), range=[start, end]), yaxis_title='km/h',
                                      legend=dict(orientation='h', y=1.05))
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("(Optional) Seasonal/Trend Decomposition — Single Axis Control (Synced Panes)")
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
                help=("STL (robust): Separates trend/seasonality using LOESS smoothing, less sensitive to outliers; "
                      "Seasonal Decompose (additive): Assumes a fixed seasonal shape, faster to compute.")
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

                # ---- 控制列 ----
                c_roll1, c_roll2, c_roll3 = st.columns([1, 1, 2])
                max_win = int(min(168, n))  # 最多 7 天
                roll_h = c_roll1.slider("Residual rolling window (hours)", 3, max_win, min(24, max_win), 1)
                roll_stat = c_roll2.selectbox("Aggregation", ["mean", "sum", "std", "abs_sum"], index=0)
                bottom_mode = c_roll3.radio("Bottom Panel Content", ["Seasonal", "Residual", "Seasonal & Residual"],
                                            index=2, horizontal=True)

                resid_roll = (resid.abs().rolling(roll_h, min_periods=1).sum()
                              if roll_stat == "abs_sum"
                              else getattr(resid.rolling(roll_h, min_periods=1), roll_stat)())

                # ---- 三排：上/中/下（中排只要 bar）----
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    specs=[[{"type": "xy"}], [{"type": "bar"}], [{"type": "xy"}]],
                    vertical_spacing=0.10,
                    row_heights=[0.52, 0.18, 0.30],
                    subplot_titles=(
                        f"Observed & Trend — {decomp_year}-{month_num:02d}",
                        f"Residual (rolling {roll_stat}, {roll_h}h) — Navigator",
                        "Seasonal / Residual (select above)"
                    )
                )

                # Row 1：Observed + Trend（線）
                fig.add_trace(go.Scatter(x=obs.index, y=obs.values, name="Observed", line=dict(width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=trend.index, y=trend.values, name="Trend", line=dict(width=3)), row=1, col=1)

                # Row 2：只放 rolling bar（不加任何其他元素/線/shape/legend）
                fig.add_trace(
                     go.Scatter(
                        x=resid_roll.index, 
                        y=resid_roll.values, 
                        fill='tozeroy',  mode='none',     
                        showlegend=False,fillcolor='rgba(255, 82, 82, 0.5)' ),row=2, col=1
                )

                # Row 3：Seasonal / Residual（線）
                if bottom_mode in ["Seasonal", "Seasonal & Residual"]:
                    fig.add_trace(go.Scatter(x=seas.index, y=seas.values, name="Seasonal", line=dict(width=1.5)), row=3, col=1)
                if bottom_mode in ["Residual", "Seasonal & Residual"]:
                    fig.add_trace(go.Scatter(x=resid.index, y=resid.values, name="Residual", line=dict(width=1)), row=3, col=1)

                # 同步 x；rangeslider 只在中排
                # fig.update_xaxes(matches='x')
                # fig.update_xaxes(rangeslider=dict(visible=True), row=3, col=1)


                fig.update_layout(
                    height=740, margin=dict(t=110, b=60, l=70, r=30),
                    yaxis_title="veh/h", bargap=0.02, uirevision="stl_decomp_sync"
                )

                # 調整標題外觀
                if hasattr(fig.layout, "annotations") and fig.layout.annotations:
                    for ann in fig.layout.annotations:
                        ann.y += 0.05
                        ann.font.size = 14
                        # ann.bgcolor = "rgba(30,30,30,0.6)" # Removed for light theme
                        ann.borderpad = 4

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough hourly points in this month/year (need ≥ 48).")
        else:
            st.info("No rows for the selected month.")

# ---------- 空間分析 (Spatial) ----------
with TAB_SPATIAL:
    st.subheader("Map Hotspots + Regional Summary")
    
    # Check for necessary columns
    if {'latitude', 'longitude', 'traffic_volume'}.issubset(_df.columns):
        
        # --- 新增：地圖類型選擇 ---
        map_type = st.radio(
            "Select Map Type",
            ["Heatmap", "Scatterplot"],
            index=0,
            horizontal=True,
            help=(
                "**Heatmap**: Shows the density of traffic volume in an area. Good for identifying general hotspots.\n\n"
                "**Scatterplot**: Shows each individual data point, colored by its traffic volume. Good for detailed analysis."
            )
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            # Filter out data with no lat/lon for the map to prevent errors
            map_df = _df[['longitude', 'latitude', 'traffic_volume']].dropna()
            
            # --- 動態選擇圖層 ---
            if map_type == "Heatmap":
                layer = pdk.Layer(
                    'HeatmapLayer',
                    data=map_df,
                    get_position='[longitude, latitude]',
                    get_weight='traffic_volume',
                    opacity=0.8,
                    pickable=False
                )
                tooltip = None
            else: # Scatterplot
                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=map_df,
                    get_position='[longitude, latitude]',
                    get_fill_color='[255, (1 - traffic_volume / 10000) * 255, 0, 140]', # Color by volume
                    get_radius='(traffic_volume / 80) + 50', # Size by volume (Increased)
                    pickable=True,
                    radius_min_pixels=3, # Increased min size
                    radius_max_pixels=100,
                )
                tooltip = {
                    "html": "<b>Traffic Volume:</b> {traffic_volume}",
                    "style": { "color": "white" }
                }

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=map_df['latitude'].mean(),
                    longitude=map_df['longitude'].mean(),
                    zoom=9,
                    pitch=45 if map_type == "Scatterplot" else 0, # Heatmap is better in 2D
                ),
                layers=[layer],
                tooltip=tooltip
            ))
        with c2:
            st.info("""
            **Map Legend**
            - **Heatmap**: Shows the geographical concentration of traffic volume.
            - **Height**: Taller bars indicate higher total volume in that area.
            - **Interaction**: You can zoom, pan, and rotate the map to view details.
            """)
            # --- Metrics moved here to ensure they reflect all filters ---
            if 'region_name' in _df.columns:
                st.metric("Number of Regions Covered", _df['region_name'].nunique())
            st.metric("Geographical Points", f"{len(_df[['latitude', 'longitude']].dropna()):,}")

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

# ---------- 趨勢分析 (Trends) - Redesigned ----------
with TAB_TREND:
    st.subheader("📈 Daily Trends & Weekly Pattern Analysis")

    # ----- 互動式選項 -----
    c1, c2 = st.columns([1, 2])
    # 讓使用者選擇要分析的指標
    selected_metric = c1.radio(
        "Select Metric",
        ['traffic_volume', 'average_speed'],
        format_func=lambda x: 'Volume' if x == 'traffic_volume' else 'Speed',
        horizontal=True
    )
    
    # 根據選擇的指標設定圖表標題
    metric_label = "Median Volume" if selected_metric == 'traffic_volume' else "Median Speed"
    yaxis_title = "veh/h" if selected_metric == 'traffic_volume' else "km/h"

    st.markdown("---")

    # ----- 圖表一：每日流量/速度趨勢圖 (週間 vs. 週末) -----
    st.subheader(f"Weekday vs. Weekend {metric_label} Variation")
    
    if {'dow', 'hour', selected_metric}.issubset(_df.columns):
        # 1. 建立一個新欄位來區分工作日與週末
        df_trend = _df.copy()
        df_trend['day_type'] = df_trend['dow'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

        # 2. 進行分組計算
        daily_pattern = df_trend.groupby(['day_type', 'hour'])[selected_metric].median().reset_index()

        # 3. 繪製折線圖
        fig_line = px.line(
            daily_pattern,
            x='hour',
            y=selected_metric,
            color='day_type',  # 用顏色區分工作日和週末
            markers=True,
            labels={'hour': 'Hour', selected_metric: yaxis_title, 'day_type': 'Day Type'},
            template='plotly_white'
        )
        
        # 4. 新增上午和下午的尖峰標示線
        fig_line.add_vline(x=8, line_width=2, line_dash="dash", line_color="grey", annotation_text="AM Peak")
        fig_line.add_vline(x=17, line_width=2, line_dash="dash", line_color="grey", annotation_text="PM Peak")
        
        fig_line.update_layout(
            height=500,
            title_text='Daily Trend Comparison: Weekday vs. Weekend',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Missing 'dow', 'hour', or selected metric columns, cannot draw daily trend chart.")

    st.markdown("---")
    
    # ----- 圖表二：優化版熱力圖 -----
    st.subheader("Enhanced Heatmap (with values and contrast colors)")

    if {'dow', 'hour', 'average_speed'}.issubset(_df.columns):
        pivot = _df.pivot_table(index='dow', columns='hour', values='average_speed', aggfunc='median')
        pivot.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][:len(pivot.index)]
        
        # 使用更有意義的顏色，並顯示數值
        fig_heatmap = px.imshow(
            pivot,
            text_auto=True,  # 在格子上顯示數值
            aspect='auto',
            color_continuous_scale=px.colors.diverging.RdYlGn, # 使用 紅-黃-綠 色階 (低速紅, 高速綠)
            labels=dict(color="Median Speed (km/h)", x="Hour", y="Day of Week"),
            template='plotly_white'
        )
        
        fig_heatmap.update_traces(textfont_size=10) # 調整格子內字體大小
        fig_heatmap.update_layout(
            height=520,
            title='Hourly Median Speed Heatmap'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Missing 'dow', 'hour', or 'average_speed' columns, cannot draw heatmap.")
