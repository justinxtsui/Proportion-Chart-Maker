import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title('Proportional (100%) Stacked Bar Chart Generator')
st.markdown('Upload a CSV or Excel, select a main x-axis category, and select how the bar is grouped (split) within each x category. Example: Year (X), Grant Type (Split).')

uploaded_file = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx'])
df = None
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

if df is not None:
    st.write('First few rows of your data:')
    st.dataframe(df.head())
    x_axis_col = st.selectbox('Select Main Category for X-Axis (e.g., Year)', df.columns)
    group_col = st.selectbox(
        'Select Category to Group Within Each X (e.g., Type of Grant)',
        [c for c in df.columns if c != x_axis_col]
    )
    
    use_sum = st.toggle('Use sum of values instead of count of rows', value=False)
    
    if use_sum:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_col = st.selectbox(
            'Select column to sum:',
            [c for c in numeric_cols if c not in [x_axis_col, group_col]]
        )
        st.write(f"You selected X-axis: {x_axis_col}, Group/Split: {group_col}, Value: Sum of {value_col}")
    else:
        value_col = None
        st.write(f"You selected X-axis: {x_axis_col}, Group/Split: {group_col}, Value: Count of rows")

    plot_df = df.copy()
    plot_df[x_axis_col] = plot_df[x_axis_col].astype(str)
    plot_df[group_col] = plot_df[group_col].astype(str)
    
    if use_sum and value_col:
        grouped = plot_df.groupby([x_axis_col, group_col])[value_col].sum().reset_index()
        value_label = f'Sum of {value_col}'
    else:
        grouped = plot_df.groupby([x_axis_col, group_col]).size().reset_index(name='count')
        value_col = 'count'
        value_label = 'Count'
    
    total = grouped.groupby(x_axis_col)[value_col].transform('sum')
    grouped['proportion'] = 100 * grouped[value_col] / total
    pivot_df = grouped.pivot(index=x_axis_col, columns=group_col, values='proportion').fillna(0)
    
    try:
        pivot_df.index = pd.to_numeric(pivot_df.index)
        pivot_df = pivot_df.sort_index()
    except:
        pivot_df = pivot_df.sort_index()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#EDD9E4', '#6F2A58', '#1B1B1B', '#CCCCCC', '#B1B2FF', '#FFDAB9', '#BEE7B8', '#5679A6', '#FFE156']
    for i, seg in enumerate(pivot_df.columns):
        if i == 0:
            bottom = None
        else:
            col_ordered = pivot_df.columns[:i]
            bottom = pivot_df[list(col_ordered)].sum(axis=1)
        color = colors[i % len(colors)]
        ax.bar(pivot_df.index, pivot_df[seg], label=str(seg), bottom=bottom, color=color)
    ax.set_ylabel('Percentage (%)', fontfamily='Public Sans', fontsize=12)
    ax.set_xlabel(x_axis_col, fontfamily='Public Sans', fontsize=12)
    ax.set_xticks(range(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.index, fontfamily='Public Sans', fontsize=12)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16, frameon=False, prop={'family': 'Public Sans'}, labelspacing=1.2)
    ax.set_title(f'Proportional Breakdown by {x_axis_col} and {group_col} ({value_label})', fontfamily='Public Sans', fontsize=16)
    st.pyplot(fig)
else:
    st.info('Upload a file first to get started.')
