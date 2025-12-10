import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('Proportional (100%) Stacked Bar Chart Generator')

uploaded_file = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write('Preview of your data:')
    st.dataframe(df.head())
    
    x_col = st.selectbox('X-Axis Category', df.columns)
    group_col = st.selectbox('Group By', [c for c in df.columns if c != x_col])
    
    use_sum = st.toggle('Sum values instead of count rows')
    
    if use_sum:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        val_col = st.selectbox('Column to sum', [c for c in num_cols if c not in [x_col, group_col]])
        data = df.groupby([x_col, group_col])[val_col].sum().reset_index(name='value')
    else:
        data = df.groupby([x_col, group_col]).size().reset_index(name='value')
    
    data['total'] = data.groupby(x_col)['value'].transform('sum')
    data['pct'] = 100 * data['value'] / data['total']
    
    pivot = data.pivot(index=x_col, columns=group_col, values='pct').fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#EDD9E4', '#6F2A58', '#1B1B1B', '#CCCCCC', '#B1B2FF']
    
    bottom = None
    for i, col in enumerate(pivot.columns):
        ax.bar(range(len(pivot)), pivot[col], bottom=bottom, 
               label=col, color=colors[i % len(colors)])
        bottom = pivot[col] if bottom is None else bottom + pivot[col]
    
    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index, rotation=45)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
else:
    st.info('Upload a file to get started')
