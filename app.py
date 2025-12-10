import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io # Added for download button

# --- New Helper Functions from Ranklin ---

# @st.cache_data is added here for optimization!
@st.cache_data
def read_any_table(file):
    """Robustly read CSV or Excel files, handling sheets and library requirements."""
    name = getattr(file, "name", "") or ""
    ext = os.path.splitext(name)[1].lower()

    if ext in [".xlsx", ".xls"]:
        # Simplified library check for deployment purposes (real code would use try/except)
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        try:
            xls = pd.ExcelFile(file, engine=engine)
            # Use a container for selectbox to keep app cleaner
            with st.container():
                sheet = st.selectbox("Select sheet:", xls.sheet_names, index=0)
            return pd.read_excel(file, sheet_name=sheet, engine=engine)
        except Exception as e:
            st.error(f"Error reading Excel: {e}. Ensure you have 'openpyxl'/'xlrd' installed.")
            st.stop()
            
    # CSV fallback
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="latin-1")

def apply_category_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Let the user choose a column and filter by selected categories."""
    with st.expander("📊 Optional Data Filter", expanded=False):
        cols = [str(c) for c in df.columns]
        if not cols: return df

        filter_col = st.selectbox("Filter column:", options=cols, index=0)
        ser_disp = df[filter_col].astype(str).fillna("").replace({"nan": "", "None": ""})
        
        uniques = pd.Series(ser_disp.unique(), dtype=str).fillna("")
        display_vals = uniques.replace({"": "(blank)"})
        display_vals = sorted(display_vals.tolist(), key=lambda x: x.lower())

        # Limit display for performance
        if len(display_vals) > 500:
             st.warning("Too many unique values — showing only the top 500.")
             display_vals = display_vals[:500]
        
        selected_vals = st.multiselect("Select categories:", options=display_vals)
        mode = st.radio("Filter mode:", ["Include", "Exclude"], horizontal=True)

        if selected_vals:
            selected_raw = [("" if v == "(blank)" else v) for v in selected_vals]
            mask = ser_disp.isin(selected_raw)
            df_out = df[mask] if mode == "Include" else df[~mask]
            st.success(f"Filtered to {len(df_out):,} rows (from {len(df):,} total).")
            return df_out

    return df

def int_commas(n):
    """Format integers with commas."""
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

# --- Original App Logic Starts Here ---

st.set_page_config(layout="wide")
st.title('Proportional (100%) Stacked Bar Chart Generator')
st.markdown('Upload a CSV or Excel, select a main x-axis category, and select how the bar is grouped (split) within each x category.')

uploaded_file = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx', 'xls'])
df = None

if uploaded_file:
    df = read_any_table(uploaded_file)
    
    if df is not None:
        # Apply filter before showing options
        df = apply_category_filter(df)
        
        st.write('First few rows of your data:')
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis_col = st.selectbox('Select Main Category for X-Axis (e.g., Year)', df.columns)
        
        with col2:
            group_col = st.selectbox(
                'Select Category to Group Within Each X (e.g., Type of Grant)',
                [c for c in df.columns if c != x_axis_col]
            )
        
        st.markdown('---')
        
        col_sum, col_val = st.columns([1, 2])
        
        with col_sum:
            use_sum = st.toggle('Use sum of values instead of count of rows', value=False)
        
        value_col = None
        if use_sum:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            with col_val:
                value_col = st.selectbox(
                    'Select column to sum:',
                    [c for c in numeric_cols if c not in [x_axis_col, group_col]]
                )
            value_label = f'Sum of {value_col}'
        else:
            value_label = 'Count of rows'

        st.info(f"Summary: X-axis: **{x_axis_col}**, Group/Split: **{group_col}**, Value: **{value_label}**")

        # --- Data Transformation (Cached for performance) ---
        @st.cache_data
        def calculate_proportions(data_frame, x_col, group_col, use_sum_flag, val_col=None):
            plot_df = data_frame.copy()
            # Ensure categorical columns are treated as strings
            plot_df[x_col] = plot_df[x_col].astype(str)
            plot_df[group_col] = plot_df[group_col].astype(str)
            
            if use_sum_flag and val_col:
                grouped = plot_df.groupby([x_col, group_col])[val_col].sum().reset_index()
                val_col_name = val_col
            else:
                grouped = plot_df.groupby([x_col, group_col]).size().reset_index(name='count')
                val_col_name = 'count'
            
            # Calculate total for each X-axis category
            total = grouped.groupby(x_col)[val_col_name].transform('sum')
            grouped['proportion'] = 100 * grouped[val_col_name] / total
            
            # Create the pivot table for plotting
            pivot_df = grouped.pivot(index=x_col, columns=group_col, values='proportion').fillna(0)
            
            # Try to sort numerically if the index looks like numbers (e.g., Years)
            try:
                pivot_df.index = pd.to_numeric(pivot_df.index)
                pivot_df = pivot_df.sort_index()
            except:
                # Fallback to string sort
                pivot_df = pivot_df.sort_index()
                
            return pivot_df

        if (use_sum and value_col) or (not use_sum):
            pivot_df = calculate_proportions(df, x_axis_col, group_col, use_sum, value_col)

            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Use custom colors (can be moved to a helper/constant)
            colors = ['#6F2A58', '#1B1B1B', '#EDD9E4', '#CCCCCC', '#B1B2FF', '#FFDAB9', '#BEE7B8', '#5679A6', '#FFE156']
            
            # Matplotlib styling settings
            plt.rcParams['font.family'] = 'Public Sans'
            
            bottom = None
            for i, seg in enumerate(pivot_df.columns):
                color = colors[i % len(colors)]
                ax.bar(pivot_df.index.astype(str), pivot_df[seg], label=str(seg), bottom=bottom, color=color)
                
                # Update bottom for the next stack
                if bottom is None:
                    bottom = pivot_df[seg]
                else:
                    bottom = bottom + pivot_df[seg]
            
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_xlabel(x_axis_col, fontsize=12)
            
            # Clean up plot for better data-ink ratio
            ax.set_ylim(0, 100)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False) # Turn off grid lines
            
            ax.legend(
                loc='upper left', 
                bbox_to_anchor=(1.05, 1), 
                fontsize=11, 
                frameon=False, 
                labelspacing=1.2,
                title=group_col
            )
            
            ax.set_title(
                f'Proportional Breakdown by {x_axis_col} and {group_col} ({value_label})', 
                fontsize=16
            )
            
            st.pyplot(fig)
            
            # Download button (using io buffer from Ranklin)
            st.markdown("---")
            svg_buffer = io.BytesIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
            svg_buffer.seek(0)
            st.download_button(
                label="Download Chart as SVG",
                data=svg_buffer,
                file_name=f"proportional_stacked_bar_{x_axis_col}_{group_col}.svg",
                mime="image/svg+xml",
            )
        else:
            st.warning("Please select a valid numeric column to sum.")
            
else:
    st.info('Upload a file first to get started.')
