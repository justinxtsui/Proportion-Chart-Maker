import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm # NEW IMPORT for font handling
import os # NEW IMPORT for file path handling

# --- FONT REGISTRATION (The Solution) ---
# 1. Define the font file path (assuming it's in the same directory)
FONT_PATH = 'PublicSans-Regular.ttf' 

# 2. Check if the font file exists
if os.path.exists(FONT_PATH):
    # 3. Register the font file with Matplotlib
    fm.fontManager.addfont(FONT_PATH)
    
    # 4. Define the family name Matplotlib will use (usually the internal name)
    # We must check the actual font name after registration
    # If the registration is successful, we can try to use it
    
    # We'll use the filename's name as a proxy, but Matplotlib often knows it
    PUBLIC_SANS_NAME = 'Public Sans' # This is the name Matplotlib should use

    # Set the global default font to Public Sans
    # This cleans up the code below, as we don't need fontfamily='Public Sans' everywhere
    plt.rcParams['font.family'] = PUBLIC_SANS_NAME

    st.success(f"Successfully loaded and set '{PUBLIC_SANS_NAME}' font.")
else:
    # Fallback if the file is missing (good practice for deployment)
    plt.rcParams['font.family'] = 'sans-serif'
    st.warning(f"Font file '{FONT_PATH}' not found. Using system 'sans-serif' as fallback.")
# ------------------------------------------

st.set_page_config(layout="wide")
st.title('Proportional (100%) Stacked Bar Chart Generator')
st.markdown('Upload a CSV or Excel, select a main x-axis category, and select how the bar is grouped (split) within each x category. Example: Year (X), Grant Type (Split).')

uploaded_file = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx'])
# ... (The rest of your code remains the same as your original, 
# but you can now remove all the `fontfamily='Public Sans'` arguments, 
# as the font is set globally with plt.rcParams.
# I'll include the plotting part to show the cleaned-up code.)

df = None
if uploaded_file:
    # Caching data for performance improvement
    @st.cache_data
    def load_data(file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)

    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")

if df is not None:
    st.write('First few rows of your data:')
    st.dataframe(df.head())
    
    # Use st.empty to ensure widget values are only set once
    with st.container():
        x_axis_col = st.selectbox('Select Main Category for X-Axis (e.g., Year)', df.columns)
        group_col = st.selectbox(
            'Select Category to Group Within Each X (e.g., Type of Grant)',
            [c for c in df.columns if c != x_axis_col]
        )
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_other_cols = [c for c in df.columns if c not in [x_axis_col, group_col]]
        default_value_col = numeric_cols[0] if numeric_cols else all_other_cols[0] if all_other_cols else df.columns[0]
        
        try:
            default_index = all_other_cols.index(default_value_col)
        except ValueError:
            default_index = 0
            
        value_col = st.selectbox(
            'Which column to use for the size of each segment (sum/count, e.g. Grant Amount, or Count)?',
            all_other_cols,
            index=default_index
        )
    st.write(f"You selected X-axis: **{x_axis_col}**, Group/Split: **{group_col}**, Value: **{value_col}**")

    # --- Data Processing (same as before) ---
    plot_df = df.copy()
    plot_df[x_axis_col] = plot_df[x_axis_col].astype(str)
    plot_df[group_col] = plot_df[group_col].astype(str)
    
    if pd.api.types.is_numeric_dtype(plot_df[value_col]):
        grouped = plot_df.groupby([x_axis_col, group_col])[value_col].sum().reset_index()
    else:
        grouped = plot_df.groupby([x_axis_col, group_col]).size().reset_index(name='count')
        value_col = 'count'
    
    total = grouped.groupby(x_axis_col)[value_col].transform('sum')
    grouped['proportion'] = 100 * grouped[value_col] / total
    pivot_df = grouped.pivot(index=x_axis_col, columns=group_col, values='proportion').fillna(0)
    
    try:
        pivot_df.index = pd.to_numeric(pivot_df.index)
        pivot_df = pivot_df.sort_index()
        pivot_df.index = pivot_df.index.astype(str)
    except:
        pivot_df = pivot_df.sort_index()

    # --- Matplotlib Plotting (Cleaned up: removed fontfamily) ---
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#EDD9E4', '#6F2A58', '#1B1B1B', '#CCCCCC', '#B1B2FF', '#FFDAB9', '#BEE7B8', '#5679A6', '#FFE156']
    
    bottom = np.zeros(len(pivot_df.index))
    
    for i, seg in enumerate(pivot_df.columns):
        color = colors[i % len(colors)]
        ax.bar(pivot_df.index, pivot_df[seg], label=str(seg), bottom=bottom, color=color)
        bottom += pivot_df[seg]

    # All fontfamily arguments are gone or simplified because of plt.rcParams
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel(x_axis_col, fontsize=12)
    ax.set_xticks(range(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.index, fontsize=12)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16, frameon=False, labelspacing=1.2)
    ax.set_title(f'Proportional Breakdown of {value_col} by {x_axis_col} and {group_col}', fontsize=16)
    st.pyplot(fig)
else:
    st.info('Upload a file first to get started.')
