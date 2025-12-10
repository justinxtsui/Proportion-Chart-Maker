import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Set Page Configuration ---
st.set_page_config(layout="wide")
st.title('Proportional (100%) Stacked Bar Chart Generator')
st.markdown('Upload a CSV or Excel file, select your X-Axis category, and the field to group/split the bars.')

# --- Caching Functions for Performance ---

@st.cache_data
def load_data(uploaded_file):
    """Caches the data loading process."""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        # Handles both .xlsx and .xls
        return pd.read_excel(uploaded_file)

@st.cache_data
def calculate_proportions(df, x_col, group_col, use_sum, val_col=None):
    """Caches the heavy data grouping and proportion calculation."""
    df_clean = df.copy()

    # 1. Group and Aggregate
    if use_sum and val_col:
        # Calculate sum of the specified column
        data = df_clean.groupby([x_col, group_col])[val_col].sum().reset_index(name='value')
        value_metric = f'Sum of {val_col}'
    else:
        # Calculate count of rows
        data = df_clean.groupby([x_col, group_col]).size().reset_index(name='value')
        value_metric = 'Count'

    # 2. Calculate Proportions
    data['total'] = data.groupby(x_col)['value'].transform('sum')
    data['pct'] = 100 * data['value'] / data['total']

    # 3. Pivot for Plotting (Index=X-Axis, Columns=Groups)
    pivot_df = data.pivot(index=x_col, columns=group_col, values='pct').fillna(0)
    
    # Attempt to sort X-axis index numerically (e.g., if X is a Year column)
    try:
        pivot_df.index = pd.to_numeric(pivot_df.index)
        pivot_df = pivot_df.sort_index()
    except Exception:
        # Fallback to alphabetical sort
        pivot_df = pivot_df.sort_index()

    return pivot_df, value_metric

# --- Main Application Logic ---

uploaded_file = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    st.subheader('1. Data Preview & Column Selection')
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox('Select Main X-Axis Category (The bars)', df.columns)
    
    with col2:
        # Ensure group_col is not the same as x_col
        group_col = st.selectbox('Select Grouping/Split Category (The segments)', 
                                 [c for c in df.columns if c != x_col])
    
    st.markdown('---')
    
    # --- Value Calculation Options ---
    st.subheader('2. Value Calculation')
    
    use_sum = st.toggle('Sum numeric column values instead of counting rows', value=False)
    val_col = None
    
    if use_sum:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not num_cols:
             st.error("No numeric columns found to sum.")
             st.stop()
        val_col = st.selectbox('Select Column to Sum:', 
                               [c for c in num_cols if c not in [x_col, group_col]])

    # --- Calculation and Plotting ---
    if use_sum and not val_col:
        st.warning("Please select a column to sum.")
    else:
        st.subheader('3. Generated Chart')
        
        # Call the cached calculation function
        pivot, value_metric = calculate_proportions(df, x_col, group_col, use_sum, val_col)

        # Plotting logic
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Custom color pallet
        colors = ['#6F2A58', '#1B1B1B', '#EDD9E4', '#CCCCCC', '#B1B2FF', '#FFDAB9', '#BEE7B8', '#5679A6']
        
        # Ensure index is treated as string for plotting if it was converted to numeric for sorting
        x_labels = pivot.index.astype(str)
        
        bottom = None
        for i, col in enumerate(pivot.columns):
            color = colors[i % len(colors)]
            
            ax.bar(x_labels, pivot[col], bottom=bottom, 
                   label=col, color=color, edgecolor='white', linewidth=0.5)
                   
            # Update the bottom for the next stack
            if bottom is None:
                bottom = pivot[col]
            else:
                bottom = bottom + pivot[col]
        
        # Title and Labels
        ax.set_title(f'Proportional Breakdown by {x_col} ({value_metric})', fontsize=16)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xlabel(x_col, fontsize=12)

        # X-tick setup
        # Use existing x_labels for cleaner display (no range() needed)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        # Axis limits and legend
        ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, title=group_col)
        
        # Clean up chart borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False) # Optional: sometimes useful to keep for a baseline
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)

else:
    st.info('Upload a file to begin generating your proportional stacked bar chart.')
