import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import plotly.graph_objects as go
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import re
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import json
import base64
import io
import time

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Enterprise Data Analytics Platform",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

#CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Hammersmith+One&display=swap');
    
    /* Light whitish background */
    .stApp { 
        font-family: 'Inter', sans-serif; 
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 50%, #f1f3f4 100%);
        min-height: 100vh; 
    }
    
    .main .block-container { 
        background: rgba(255, 255, 255, 0.98); 
        border-radius: 20px; 
        padding: 2rem; 
        backdrop-filter: blur(10px); 
        box-shadow: 0 10px 30px rgba(0,0,0,0.08); 
        margin-top: 2rem; 
        margin-bottom: 2rem; 
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Enhanced Sidebar Styling to Match Main Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%) !important;
        border-right: 2px solid rgba(102, 126, 234, 0.2) !important;
        box-shadow: 4px 0 15px rgba(0,0,0,0.1) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    
    /* Sidebar content styling */
    .css-1d391kg {
        background: transparent !important;
        padding: 1.5rem !important;
    }
    
    /* Sidebar text and headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .css-1lcbmhc {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .css-1lcbmhc {
        color: #333333 !important;
    }
    
    /* Sidebar widgets styling */
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Sidebar file uploader */
    [data-testid="stSidebar"] .stFileUploader > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] .stFileUploader > div:hover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.05) !important;
    }
    
    /* Sidebar selectbox */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Sidebar checkbox */
    [data-testid="stSidebar"] .stCheckbox {
        background: rgba(255, 255, 255, 0.7) !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stCheckbox > label {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar info box */
    [data-testid="stSidebar"] .stAlert {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #2c3e50 !important;
    }
    
    /* Sidebar success/error messages */
    [data-testid="stSidebar"] .stSuccess {
        background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%) !important;
        border: 1px solid #4caf50 !important;
        color: #2e7d32 !important;
    }
    
    [data-testid="stSidebar"] .stError {
        background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%) !important;
        border: 1px solid #f44336 !important;
        color: #c62828 !important;
    }
    
    [data-testid="stSidebar"] .stWarning {
        background: linear-gradient(135deg, #fff3e0 0%, #fef7e0 100%) !important;
        border: 1px solid #ff9800 !important;
        color: #ef6c00 !important;
    }
    
    /* Sidebar markdown styling */
    [data-testid="stSidebar"] .stMarkdown {
        color: #2c3e50 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(102, 126, 234, 0.3) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Sidebar expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 8px !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Updated header with Hammersmith One font */
    .main-header { 
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
        padding: 3rem 2rem; 
        border-radius: 20px; 
        margin-bottom: 2rem; 
        box-shadow: 0 8px 25px rgba(0,0,0,0.08); 
        color: #2c3e50; 
        text-align: center; 
        position: relative; 
        overflow: hidden; 
        animation: slideInDown 0.8s ease-out; 
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Hammersmith One font for main header with black color */
    .main-header h1 {
        font-family: 'Hammersmith One', sans-serif !important;
        font-weight: 400 !important;
        font-size: 3.5rem !important;
        letter-spacing: 0.02em !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
        margin: 0 !important;
        color: #000000 !important;
        line-height: 1.2 !important;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        font-size: 1.2rem !important;
        color: #333333 !important;
        margin: 1rem 0 0 0 !important;
        letter-spacing: 0.3px !important;
    }
    
    .main-header::before { 
        content: ''; 
        position: absolute; 
        top: -50%; 
        left: -50%; 
        width: 200%; 
        height: 200%; 
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent); 
        animation: shine 3s infinite; 
    }
    
    @keyframes slideInDown { 
        from { transform: translateY(-50px); opacity: 0; } 
        to { transform: translateY(0); opacity: 1; } 
    }
    
    @keyframes shine { 
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); } 
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); } 
    }
    
    /* Light KPI cards */
    .kpi-card { 
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%); 
        border-radius: 15px; 
        padding: 2rem; 
        box-shadow: 0 5px 20px rgba(0,0,0,0.08); 
        border: 1px solid rgba(0,0,0,0.08); 
        margin-bottom: 1.5rem; 
        transition: all 0.3s ease; 
        position: relative; 
        overflow: hidden; 
        animation: slideInUp 0.6s ease-out; 
    }
    
    .kpi-card:hover { 
        transform: translateY(-3px); 
        box-shadow: 0 12px 30px rgba(0,0,0,0.12); 
    }
    
    .kpi-card::before { 
        content: ''; 
        position: absolute; 
        top: 0; 
        left: 0; 
        right: 0; 
        height: 4px; 
        background: linear-gradient(90deg, #e3f2fd, #f3e5f5); 
    }
    
    @keyframes slideInUp { 
        from { transform: translateY(30px); opacity: 0; } 
        to { transform: translateY(0); opacity: 1; } 
    }
    
    /* Light chart containers */
    .chart-container { 
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%); 
        border-radius: 15px; 
        padding: 2rem; 
        box-shadow: 0 5px 20px rgba(0,0,0,0.08); 
        margin-bottom: 2rem; 
        border: 1px solid rgba(0,0,0,0.08); 
        transition: all 0.3s ease; 
        animation: fadeIn 0.8s ease-out; 
    }
    
    .chart-container:hover { 
        box-shadow: 0 8px 25px rgba(0,0,0,0.12); 
    }
    
    @keyframes fadeIn { 
        from { opacity: 0; transform: scale(0.98); } 
        to { opacity: 1; transform: scale(1); } 
    }
    
    /* Light themed buttons */
    .stButton > button { 
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
        color: #2c3e50; 
        border: 2px solid rgba(0,0,0,0.1); 
        border-radius: 12px; 
        padding: 0.8rem 2rem; 
        font-weight: 600; 
        font-size: 1rem; 
        transition: all 0.3s ease; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.1); 
        position: relative; 
        overflow: hidden; 
    }
    
    .stButton > button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 15px rgba(0,0,0,0.15); 
        background: linear-gradient(135deg, #bbdefb 0%, #e1bee7 100%);
    }
    
    /* Light themed tabs */
    .stTabs [data-baseweb="tab-list"] { 
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
        border-radius: 15px; 
        padding: 0.5rem; 
        box-shadow: 0 3px 15px rgba(0,0,0,0.08); 
        gap: 0.5rem; 
        border: 1px solid rgba(0,0,0,0.08);
    }
    
    .stTabs [data-baseweb="tab"] { 
        background: transparent; 
        border-radius: 10px; 
        padding: 1rem 1.5rem; 
        font-weight: 600; 
        color: #2c3e50; 
        transition: all 0.3s ease; 
        border: 2px solid transparent; 
    }
    
    .stTabs [data-baseweb="tab"]:hover { 
        background: rgba(227, 242, 253, 0.5); 
        border-color: rgba(227, 242, 253, 0.8); 
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
        color: #1565c0; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.1); 
        border-color: rgba(21, 101, 192, 0.2);
    }
    
    /* Light chat container */
    .chat-container { 
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%); 
        border-radius: 15px; 
        padding: 2rem; 
        box-shadow: 0 5px 20px rgba(0,0,0,0.08); 
        margin: 1rem 0; 
        border: 1px solid rgba(0,0,0,0.08); 
    }
    
    /* Light themed metrics */
    [data-testid="metric-container"] { 
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%); 
        border: 1px solid rgba(0,0,0,0.08); 
        padding: 1rem; 
        border-radius: 12px; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.08); 
        transition: all 0.3s ease; 
    }
    
    [data-testid="metric-container"]:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 15px rgba(0,0,0,0.12); 
    }
</style>
""", unsafe_allow_html=True)



#Session State
for k, v in {
    "viz_redirect": False,
    "viz_x_axis": None,
    "viz_y_axis": None,
    "viz_chart_type": "Scatter Plot",
    "active_tab": "AI Assistant"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def clean_data(df):
    df_cleaned = df.dropna(axis=1, how='all')
    threshold = len(df) * 0.05
    df_cleaned = df_cleaned.dropna(axis=1, thresh=threshold)
    return df_cleaned

def create_column_helper(df):
    def get_column_info(col_name):
        col_data = df[col_name]
        info = {
            'name': col_name,
            'type': str(col_data.dtype),
            'non_null': col_data.count(),
            'null_count': col_data.isnull().sum(),
            'unique_values': col_data.nunique() if col_data.nunique() <= 20 else f"{col_data.nunique()} (too many to show)"
        }
        if pd.api.types.is_numeric_dtype(col_data):
            info.update({'min': col_data.min(), 'max': col_data.max(), 'mean': col_data.mean()})
        elif pd.api.types.is_object_dtype(col_data) and col_data.nunique() <= 10:
            info['sample_values'] = col_data.value_counts().head(5).to_dict()
        return info
    return get_column_info

def extract_columns_from_prompt(prompt, all_columns):
    prompt_lower = prompt.lower()
    quoted_matches = set()
    for col in all_columns:
        col_lower = col.lower()
        if (f"'{col_lower}'" in prompt_lower or
            f'"{col_lower}"' in prompt_lower or
            re.search(rf"\b{re.escape(col_lower)}\b", prompt_lower)):
            quoted_matches.add(col)
    if not quoted_matches:
        quoted_matches = {col for col in all_columns if col.lower() in prompt_lower}
    return list(quoted_matches)

def parse_visualization_request(prompt, all_columns):
    prompt_lower = prompt.lower()
    patterns = [
        r"(?:visualize|plot|graph|scatter|show|draw)\s+([a-zA-Z0-9_]+)\s+(?:vs|and|versus|against)\s+([a-zA-Z0-9_]+)",
        r"(?:visualize|plot|graph|scatter|show|draw)\s+([a-zA-Z0-9_]+)\s+and\s+([a-zA-Z0-9_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            col1, col2 = match.group(1), match.group(2)
            x_col = next((col for col in all_columns if col.lower() == col1), None)
            y_col = next((col for col in all_columns if col.lower() == col2), None)
            if x_col and y_col:
                return x_col, y_col
    return None, None

class EnhancedAnalytics:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower()]
    
    def auto_insights(self):
        insights = []
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            insights.append(f"⚠️ Dataset has {missing_data.sum():,} missing values across {(missing_data > 0).sum()} columns")
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.8)
            high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr) if x != y and x < y]
            if high_corr_pairs:
                insights.append(f"🔗 Found {len(high_corr_pairs)} highly correlated variable pairs")
        for col in self.numeric_cols[:3]:
            outliers = self.detect_outliers_isolation_forest(col)
            if len(outliers) > 0:
                pct = (len(outliers) / len(self.df)) * 100
                insights.append(f"📊 {col}: {len(outliers)} outliers detected ({pct:.1f}%)")
        return insights
    
    def detect_outliers_isolation_forest(self, column):
        if column in self.numeric_cols:
            data = self.df[[column]].dropna()
            if len(data) > 10:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(data)
                return data[outliers == -1].index.tolist()
        return []
    
    def predictive_modeling(self, target_col, feature_cols):
        try:
            data = self.df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            return {
                'model': model,
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': dict(zip(feature_cols, model.coef_)),
                'predictions': y_pred,
                'actuals': y_test
            }
        except Exception as e:
            return {'error': str(e)}
    
    def advanced_statistical_tests(self, col1, col2=None):
        results = {}
        try:
            if col1 in self.numeric_cols:
                data1 = self.df[col1].dropna()
                if len(data1) > 3:
                    sample_size = min(5000, len(data1))
                    sample_data = data1.sample(n=sample_size, random_state=42) if sample_size < len(data1) else data1
                    shapiro_stat, shapiro_p = stats.shapiro(sample_data)
                    results['normality_test'] = {'statistic': shapiro_stat, 'p_value': shapiro_p, 'sample_size': len(sample_data)}
                if col2 and col2 in self.numeric_cols:
                    data2 = self.df[col2].dropna()
                    if len(data1) > 1 and len(data2) > 1:
                        corr, p_val = stats.pearsonr(data1, data2)
                        results['pearson_correlation'] = {'correlation': corr, 'p_value': p_val}
                        spear_corr, spear_p = stats.spearmanr(data1, data2)
                        results['spearman_correlation'] = {'correlation': spear_corr, 'p_value': spear_p}
        except Exception as e:
            results['error'] = str(e)
        return results
    
    def time_series_analysis(self, date_col, value_col):
        if date_col in self.date_cols and value_col in self.numeric_cols:
            try:
                df_ts = self.df[[date_col, value_col]].dropna()
                if len(df_ts) < 2:
                    return {'error': 'Insufficient data for time series analysis'}
                df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                df_ts = df_ts.sort_values(date_col).set_index(date_col)
                df_ts['trend'] = np.arange(len(df_ts))
                slope, intercept, r_value, p_value, std_err = stats.linregress(df_ts['trend'], df_ts[value_col])
                results = {
                    'trend_slope': slope,
                    'trend_r2': r_value**2,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                    'trend_strength': 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.3 else 'weak',
                    'data_points': len(df_ts)
                }
                return results
            except Exception as e:
                return {'error': f"Time series analysis failed: {str(e)}"}
        return {'error': 'Invalid date or value column'}

class EnhancedVisualizations:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def create_dashboard_summary(self):
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Data Distribution', 'Missing Data', 'Data Types', 'Column Info'),
                specs=[[{"type": "histogram"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "bar"}]]
            )
            if self.numeric_cols:
                fig.add_trace(
                    go.Histogram(x=self.df[self.numeric_cols[0]], name="Distribution", marker_color='#667eea'),
                    row=1, col=1
                )
            missing = self.df.isnull().sum()
            if missing.sum() > 0:
                missing_top = missing.nlargest(10)
                fig.add_trace(
                    go.Bar(x=missing_top.index, y=missing_top.values, name="Missing Values", marker_color='#764ba2'),
                    row=1, col=2
                )
            dtype_counts = self.df.dtypes.value_counts()
            fig.add_trace(
                go.Pie(labels=[str(x) for x in dtype_counts.index], values=dtype_counts.values, name="Data Types"),
                row=2, col=1
            )
            col_info = pd.Series({
                'Numeric': len(self.numeric_cols),
                'Categorical': len(self.df.select_dtypes(include=['object']).columns),
                'Date': len([col for col in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[col])]),
                'Other': len(self.df.columns) - len(self.numeric_cols) - len(self.df.select_dtypes(include=['object']).columns)
            })
            fig.add_trace(
                go.Bar(x=col_info.index, y=col_info.values, name="Column Types", marker_color='#667eea'),
                row=2, col=2
            )
            fig.update_layout(
                height=600, showlegend=False, title_text="Data Overview Dashboard",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        except Exception as e:
            if self.numeric_cols:
                fig = px.histogram(self.df, x=self.numeric_cols[0], title="Data Distribution")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                return fig
            else:
                fig = go.Figure()
                fig.add_annotation(text="No numeric data available for visualization", xref="paper", yref="paper", x=0.5, y=0.5)
                return fig

def create_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'''
    <div style="margin-bottom: 10px;">  <!-- Add vertical spacing -->
        <a href="data:file/csv;base64,{b64}" download="{filename}"
        style="
            background: linear-gradient(0deg, #1E7C79 0%, #5A8298 100%);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            border: 2px solid #BCCCDC;
            display: inline-block;
        ">
        📥 Download {filename}
        </a>
    </div>
    '''
    return href

#Sidebar
with st.sidebar:
    st.markdown("# 📊 Enterprise Analytics")
    st.markdown("---")
    api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key for AI features")
    uploaded_files = st.file_uploader("Upload Data Files", type=["csv", "xlsx", "json"], accept_multiple_files=True, help="Support for CSV, Excel, and JSON files")
    if uploaded_files:
        enable_filters = st.checkbox("Enable Real-time Filtering")
    else:
        enable_filters = False
    chart_theme = st.selectbox("Chart Theme", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"], index=0)
    auto_refresh = st.checkbox("Auto-refresh Insights", value=True)
    st.markdown("---")
    st.info("💡 Enterprise-grade analytics with AI assistance")

if not uploaded_files:
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">📊 Enterprise Data Analytics Platform</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Advanced Analytics • AI-Powered Insights • Real-time Dashboards</p>
    </div>
    """, unsafe_allow_html=True)
    st.warning("⚠️ Please upload data files to begin analysis")
    st.stop()

dfs = []
for file in uploaded_files:
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        df_cleaned = clean_data(df)
        if df_cleaned.empty:
            st.warning(f"⚠️ {file.name}: No valid data after cleaning")
            continue
        dfs.append(df_cleaned)
        st.sidebar.success(f"✅ Loaded {file.name}")
    except Exception as e:
        st.error(f"Error loading {file.name}: {str(e)}")

if not dfs:
    st.error("No valid data files loaded")
    st.stop()

combined_df = pd.concat(dfs, ignore_index=True, sort=False)
combined_df = combined_df.dropna(axis=1, how='all')

analytics = EnhancedAnalytics(combined_df)
visualizations = EnhancedVisualizations(combined_df)

st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">📊 Enterprise Data Analytics Dashboard</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Professional Analytics Platform with Advanced AI Capabilities</p>
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    with st.expander("🤖 AI-Generated Insights", expanded=True):
        with st.spinner("Generating insights..."):
            insights = analytics.auto_insights()
            if insights:
                for insight in insights:
                    st.markdown(f"• {insight}")
            else:
                st.success("✅ No major data issues detected - your data looks great!")

st.markdown("## 📋 Executive Summary")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f'''
    <div class="kpi-card">
        <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem;">
            {len(combined_df):,}
        </div>
        <div style="font-size: 0.9rem; font-weight: 600; color: #666;">
            Total Records
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="kpi-card">
        <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem;">
            {len(dfs)}
        </div>
        <div style="font-size: 0.9rem; font-weight: 600; color: #666;">
            Data Sources
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
    <div class="kpi-card">
        <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem;">
            {len(combined_df.columns)}
        </div>
        <div style="font-size: 0.9rem; font-weight: 600; color: #666;">
            Total Columns
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown(f'''
    <div class="kpi-card">
        <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem;">
            {len(analytics.numeric_cols)}
        </div>
        <div style="font-size: 0.9rem; font-weight: 600; color: #666;">
            Numeric Fields
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col5:
    missing_pct = (combined_df.isnull().sum().sum() / (len(combined_df) * len(combined_df.columns)) * 100)
    st.markdown(f'''
    <div class="kpi-card">
        <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem;">
            {100-missing_pct:.1f}%
        </div>
        <div style="font-size: 0.9rem; font-weight: 600; color: #666;">
            Data Quality
        </div>
    </div>
    ''', unsafe_allow_html=True)

#Analytics Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Smart Visualizations", 
    "🔍 Advanced Statistics", 
    "🎯 Predictive Models",
    "🔗 Relationship Mining",
    "📈 Time Series",
    "🤖 AI Assistant",
    "📋 Export & Reports"
])

#Tab 1: Smart Visualizations
with tab1:
    st.markdown("### 📊 Interactive Visualizations")
    
    if st.session_state.viz_redirect:
        x_axis_default = st.session_state.viz_x_axis if st.session_state.viz_x_axis in combined_df.columns else combined_df.columns[0]
        y_axis_default = st.session_state.viz_y_axis if st.session_state.viz_y_axis in analytics.numeric_cols else analytics.numeric_cols[0]
        chart_type_default = st.session_state.viz_chart_type or "Scatter Plot"
        st.session_state.viz_redirect = False
        st.session_state.viz_x_axis = None
        st.session_state.viz_y_axis = None
        st.session_state.viz_chart_type = None
    else:
        x_axis_default = combined_df.columns[0]
        y_axis_default = analytics.numeric_cols[0]
        chart_type_default = "Scatter Plot"

    x_axis = st.selectbox("X-Axis", combined_df.columns, index=list(combined_df.columns).index(x_axis_default), key="viz_x_selectbox")
    y_axis = st.selectbox("Y-Axis", analytics.numeric_cols, index=list(analytics.numeric_cols).index(y_axis_default), key="viz_y_selectbox")
    chart_type = st.selectbox(
        "Visualization Type",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Histogram", "Correlation Heatmap", "Violin Plot"],
        index=["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Histogram", "Correlation Heatmap", "Violin Plot"].index(chart_type_default),
        key="viz_chart_type_selectbox"
    )

    col1, col2 = st.columns(2)
    with col1:
        filters = {}
        if enable_filters:
            filter_col = st.selectbox("Filter Column", [None] + list(combined_df.columns), key="filter_col")
            if filter_col:
                unique_vals = combined_df[filter_col].unique()[:50]
                selected_vals = st.multiselect(f"Select {filter_col} values", unique_vals, key="filter_vals")
                if selected_vals:
                    filters[filter_col] = selected_vals
    with col2:
        color_by = st.selectbox("Color By", [None] + list(combined_df.columns), key="color_by")
        size_by = st.selectbox("Size By", [None] + analytics.numeric_cols, key="size_by")

    #plot visualization
    if x_axis and y_axis:
        try:
            df_viz = combined_df.copy()
            if filters:
                for col, values in filters.items():
                    if values:
                        df_viz = df_viz[df_viz[col].isin(values)]
            if len(df_viz) == 0:
                st.warning("No data available after applying filters.")
            else:
                fig = None
                if chart_type == "Scatter Plot":
                    fig = px.scatter(df_viz, x=x_axis, y=y_axis, color=color_by, size=size_by,
                                   title=f"Scatter Plot: {y_axis} vs {x_axis}", template=chart_theme)
                elif chart_type == "Line Chart":
                    fig = px.line(df_viz, x=x_axis, y=y_axis, color=color_by,
                                title=f"Line Chart: {y_axis} vs {x_axis}", template=chart_theme)
                elif chart_type == "Bar Chart":
                    fig = px.bar(df_viz, x=x_axis, y=y_axis, title=f"Bar Chart: {y_axis} by {x_axis}")
                elif chart_type == "Box Plot":
                    fig = px.box(df_viz, x=x_axis, y=y_axis, color=color_by,
                               title=f"Box Plot: {y_axis} by {x_axis}", template=chart_theme)
                elif chart_type == "Histogram":
                    fig = px.histogram(df_viz, x=x_axis, color=color_by,
                                     title=f"Histogram: {x_axis}", template=chart_theme)
                elif chart_type == "Correlation Heatmap":
                    if len(analytics.numeric_cols) > 1:
                        corr_matrix = df_viz[analytics.numeric_cols].corr()
                        fig = px.imshow(corr_matrix, title="Correlation Heatmap",
                                      color_continuous_scale="RdBu", aspect="auto")
                    else:
                        st.warning("Need at least 2 numeric columns for correlation heatmap")
                elif chart_type == "Violin Plot":
                    fig = px.violin(df_viz, x=x_axis, y=y_axis, color=color_by,
                                  title=f"Violin Plot: {y_axis} by {x_axis}", template=chart_theme)

                if fig:
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"📊 Showing {len(df_viz):,} data points")
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
    else:
        st.info("👆 Please select X-axis and Y-axis columns to generate visualization")

    #data overview dashboard
    st.markdown("---") 
    st.markdown("### 📊 Data Overview Dashboard")
    try:
        overview_fig = visualizations.create_dashboard_summary()
        st.plotly_chart(overview_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating dashboard overview: {e}")



#Tab 2: Advanced Statistics
with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🔍 Advanced Statistical Analysis")
    if analytics.numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Statistical Tests")
            test_col1 = st.selectbox("Variable 1", analytics.numeric_cols, key="test1")
            test_col2 = st.selectbox("Variable 2", analytics.numeric_cols, key="test2")
            if st.button("Run Statistical Tests"):
                with st.spinner("Running statistical tests..."):
                    try:
                        test_results = analytics.advanced_statistical_tests(test_col1, test_col2)
                        if 'error' in test_results:
                            st.error(f"Error: {test_results['error']}")
                        else:
                            for test_name, result in test_results.items():
                                st.markdown(f"**{test_name.replace('_', ' ').title()}**")
                                if test_name == 'normality_test':
                                    st.metric("P-value", f"{result['p_value']:.4f}")
                                    st.metric("Sample Size", result['sample_size'])
                                    is_normal = result['p_value'] > 0.05
                                    st.success("Data appears normal" if is_normal else "Data is not normally distributed")
                                elif 'correlation' in test_name:
                                    st.metric("Correlation", f"{result['correlation']:.4f}")
                                    st.metric("P-value", f"{result['p_value']:.4f}")
                                st.markdown("---")
                    except Exception as e:
                        st.error(f"Error running tests: {str(e)}")
        with col2:
            st.markdown("#### Outlier Detection")
            outlier_col = st.selectbox("Select Column", analytics.numeric_cols, key="outlier")
            if st.button("Detect Outliers"):
                try:
                    outliers = analytics.detect_outliers_isolation_forest(outlier_col)
                    st.metric("Outliers Found", len(outliers))
                    if outliers:
                        outlier_pct = (len(outliers) / len(combined_df)) * 100
                        st.metric("Outlier Percentage", f"{outlier_pct:.2f}%")
                    fig = px.box(combined_df, y=outlier_col, title=f"Outlier Detection: {outlier_col}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error detecting outliers: {str(e)}")
    else:
        st.warning("No numeric columns available for statistical analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

#Tab 3: Predictive Modeling
with tab3:
    st.markdown("### 🎯 Predictive Modeling")
    if len(analytics.numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Model Configuration")
            target = st.selectbox("Target Variable", analytics.numeric_cols, key="target")
            features = st.multiselect("Feature Variables", [col for col in analytics.numeric_cols if col != target], key="features")
        with col2:
            if features and target:
                if st.button("🚀 Build Predictive Model"):
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)
                    results = analytics.predictive_modeling(target, features)
                    progress_bar.progress(100)
                    progress_bar.empty()
                    if 'error' not in results:
                        st.metric("R² Score", f"{results['r2_score']:.3f}")
                        st.metric("RMSE", f"{results['rmse']:.3f}")
                        importance_df = pd.DataFrame(list(results['feature_importance'].items()), columns=['Feature', 'Importance'])
                        fig = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Model Error: {results['error']}")
    else:
        st.warning("Need at least 2 numeric columns for predictive modeling.")
    st.markdown('</div>', unsafe_allow_html=True)

#Tab 4: Relationship Mining
with tab4:
    st.markdown("### 🔗 Relationship Mining")
    if len(analytics.numeric_cols) > 1:
        corr_matrix = combined_df[analytics.numeric_cols].corr()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(corr_matrix, title="Enhanced Correlation Matrix", color_continuous_scale="RdBu", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({'Variable 1': corr_matrix.columns[i], 'Variable 2': corr_matrix.columns[j], 'Correlation': corr_matrix.iloc[i, j]})
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.markdown("#### Strongest Correlations")
            st.dataframe(corr_df.head(10), use_container_width=True)
    else:
        st.warning("Need at least 2 numeric columns for relationship analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

#Tab 5: Time Series
with tab5:
    st.markdown("### 📈 Time Series Analysis")
    if analytics.date_cols and analytics.numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Date Column", analytics.date_cols, key="ts_date")
            value_col = st.selectbox("Value Column", analytics.numeric_cols, key="ts_value")
        with col2:
            if st.button("🚀 Analyze Time Series"):
                with st.spinner("Analyzing time series..."):
                    ts_results = analytics.time_series_analysis(date_col, value_col)
                    if ts_results and 'error' not in ts_results:
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        with col_metric1:
                            st.metric("Trend Strength (R²)", f"{ts_results['trend_r2']:.3f}")
                        with col_metric2:
                            st.metric("Trend Direction", ts_results['trend_direction'])
                        with col_metric3:
                            st.metric("Data Points", f"{ts_results['data_points']:,}")
                        st.info(f"📈 Trend Strength: {ts_results['trend_strength']}")
                        try:
                            fig = px.line(combined_df, x=date_col, y=value_col, title=f"Time Series: {value_col} over time")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating time series plot: {e}")
                    elif ts_results and 'error' in ts_results:
                        st.error(f"❌ {ts_results['error']}")
                    else:
                        st.warning("⚠️ Unable to perform time series analysis.")
    else:
        if not analytics.date_cols:
            st.warning("⚠️ No date columns detected in your dataset.")
        if not analytics.numeric_cols:
            st.warning("⚠️ No numeric columns available for time series analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

#Tab 6: AI Assistant
with tab6:
    if api_key:
        try:
            from langchain_community.llms import OpenAI
            from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
            
            llm = OpenAI(openai_api_key=api_key, openai_api_base="https://openrouter.ai/api/v1", model="meta-llama/llama-3-8b-instruct", temperature=0)
            col_helper = create_column_helper(combined_df)
            
            with st.expander("📋 Smart Column Reference Helper", expanded=False):
                search_term = st.text_input("🔍 Search columns:", placeholder="Type to filter columns...", key="col_search")
                column_type_filter = st.selectbox("Filter by type:", ["All", "Numeric", "Categorical", "Date"], key="type_filter")
                available_columns = list(combined_df.columns)
                if search_term:
                    available_columns = [col for col in available_columns if search_term.lower() in col.lower()]
                if column_type_filter != "All":
                    if column_type_filter == "Numeric":
                        available_columns = [col for col in available_columns if col in analytics.numeric_cols]
                    elif column_type_filter == "Categorical":
                        available_columns = [col for col in available_columns if col in analytics.categorical_cols]
                    elif column_type_filter == "Date":
                        available_columns = [col for col in available_columns if col in analytics.date_cols]
                if available_columns:
                    st.markdown(f"**Found {len(available_columns)} matching columns:**")
                    num_cols = min(3, len(available_columns))
                    col_displays = st.columns(num_cols)
                    for idx, col_name in enumerate(available_columns[:15]):
                        with col_displays[idx % num_cols]:
                            col_info = col_helper(col_name)
                            st.markdown('<div class="column-btn">', unsafe_allow_html=True)
                            if st.button(f"📌 {col_name}", key=f"col_btn_{idx}", help=f"Click to insert '{col_name}' into your query"):
                                if "user_query" not in st.session_state:
                                    st.session_state.user_query = ""
                                if st.session_state.user_query and not st.session_state.user_query.endswith(" "):
                                    st.session_state.user_query += " "
                                st.session_state.user_query += f"{col_name}"
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.caption(f"**Type:** {col_info['type']}")
                            st.caption(f"**Non-null:** {col_info['non_null']:,}")
                            if 'mean' in col_info:
                                st.caption(f"**Range:** {col_info['min']:.2f} - {col_info['max']:.2f}")
                            elif 'sample_values' in col_info:
                                st.caption(f"**Top values:** {list(col_info['sample_values'].keys())[:3]}")
                else:
                    st.info("No columns match your search criteria.")

            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            if "user_query" not in st.session_state:
                st.session_state.user_query = ""
            
            st.markdown("**💬 Chat with AI Assistant**")
            user_query = st.text_area("Ask me anything about your data...", value=st.session_state.user_query, height=100, key="chat_input_area", placeholder="Example: 'Analyze the distribution of sales_amount' or 'Compare customer_age and revenue'")
            st.session_state.user_query = user_query
            
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            send_query = st.button("🚀 Send Query", key="send_chat", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

            #Chat History
            if st.session_state.chat_messages[:-1]:
                with st.expander("🕑 Chat History", expanded=False):
                    for message in st.session_state.chat_messages[:-1]:
                        role = message.get("role", "user")
                        content = message.get("content", "")
                        st.markdown(f"**{role.capitalize()}:** {content}")

            if st.session_state.chat_messages:
                last_message = st.session_state.chat_messages[-1]
                with st.chat_message(last_message["role"]):
                    st.markdown(last_message["content"])

            if send_query and user_query.strip():
                x_col, y_col = parse_visualization_request(user_query, list(combined_df.columns))
                if x_col and y_col:
                    st.session_state.chat_messages.append({"role": "user", "content": user_query})
                    with st.chat_message("user"):
                        st.markdown(user_query)
                    with st.chat_message("assistant"):
                        st.info(f"Your question is about visualizing **{x_col}** vs **{y_col}**. Click the button below to view this visualization in the Smart Visualizations tab.")
                        st.markdown('<div class="button-container">', unsafe_allow_html=True)
                        if st.button("Go to Visualization", key="go_to_viz"):
                            st.session_state.viz_x_axis = x_col
                            st.session_state.viz_y_axis = y_col
                            st.session_state.viz_chart_type = "Scatter Plot"
                            st.session_state.viz_redirect = True
                            st.experimental_rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    mentioned_cols = extract_columns_from_prompt(user_query, list(combined_df.columns))
                    if mentioned_cols:
                        ai_df = combined_df.filter(items=mentioned_cols)
                        st.info(f"AI will use only these columns: {', '.join(mentioned_cols)}")
                    else:
                        ai_df = combined_df
                        st.info("No columns specified in prompt. AI will use the entire dataset.")

                    agent = create_pandas_dataframe_agent(
                        llm,
                        ai_df,
                        verbose=True,
                        allow_dangerous_code=True,
                        max_iterations=15,  #Reduced for better efficiency
                        max_execution_time=60.0,  #Reduced for efficiency
                        handle_parsing_errors=True,
                        early_stopping_method="force"
                    )

                    st.session_state.chat_messages.append({"role": "user", "content": user_query})
                    with st.chat_message("user"):
                        st.markdown(user_query)
                    with st.chat_message("assistant"):
                        with st.spinner("🔍 Analyzing your data..."):
                            try:
                                enhanced_prompt = (
                                    f"{user_query}\nColumns used: {', '.join(mentioned_cols) if mentioned_cols else 'all columns'}\n"
                                    "Instructions: Answer concisely in a direct, factual, and quantitative manner in just 1 to 2 sentences. "
                                    "If asked about the relationship or correlation between two columns, answer in this format: "
                                    "\"The [Column1] and [Column2] are [correlated/not correlated/weakly correlated/strongly correlated/negatively correlated/etc.]. "
                                    "The correlation coefficient is [value].\" "
                                    "Do not repeat information, do not use polite language, and do not add explanations or commentary. "
                                    "Only answer what is asked of you, and do not write anything else."
                                )
                                response = agent.run(enhanced_prompt)

                                st.markdown(response)
                                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                                st.session_state.user_query = ""
                            except Exception as e:
                                if "OUTPUT_PARSING_FAILURE" in str(e) or "parsing error" in str(e).lower():
                                    try:
                                        retry_prompt = f"Answer this question about the data: {user_query}"
                                        response = agent.run(retry_prompt)
                                        st.markdown(response)
                                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                                        st.session_state.user_query = ""
                                    except Exception as retry_error:
                                        error_msg = "I encountered a parsing error. Please try rephrasing your question more simply."
                                        st.error(error_msg)
                                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                                else:
                                    error_msg = f"I encountered an error: {str(e)}"
                                    st.error(error_msg)
                                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

        except Exception as e:
            st.error(f"Failed to initialize AI assistant: {str(e)}")
    else:
        st.warning("Please provide an API key to use the AI assistant")
    st.markdown('</div>', unsafe_allow_html=True)

#Tab 7: Export & Reports
with tab7:
    st.markdown("### 📋 Export & Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Data Export")
        csv_link = create_download_link(combined_df, "analytics_data.csv")
        st.markdown(csv_link, unsafe_allow_html=True)
        if analytics.numeric_cols:
            summary_stats = combined_df[analytics.numeric_cols].describe()
            summary_link = create_download_link(summary_stats, "summary_statistics.csv")
            st.markdown(summary_link, unsafe_allow_html=True)
    with col2:
        st.markdown("#### Generate Report")
        if st.button("📊 Generate Executive Summary"):
            report_data = {
                'Dataset Overview': {
                    'Total Records': len(combined_df),
                    'Total Columns': len(combined_df.columns),
                    'Numeric Columns': len(analytics.numeric_cols),
                    'Categorical Columns': len(analytics.categorical_cols),
                    'Missing Values': combined_df.isnull().sum().sum()
                },
                'AI Insights': analytics.auto_insights()
            }
            st.json(report_data)
    st.markdown('</div>', unsafe_allow_html=True)

#Footer
st.markdown("""
---
<div style="text-align: center; padding: 1rem 0; color: #666;">
    <strong>Enterprise Data Analytics Platform</strong> • Powered by OpenRouter AI • Built with Streamlit<br>
    <small>&copy; 2025 | Advanced Analytics & AI-Driven Insights</small>
</div>
""", unsafe_allow_html=True)
