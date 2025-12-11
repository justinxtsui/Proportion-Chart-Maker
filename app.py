import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Configuration and Setup ---
st.set_page_config(
    page_title="Dynamic Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading Function ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded CSV or Excel file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Use 'low_memory=False' to handle large files and mixed types better
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file (.csv, .xlsx, .xls).")
            return pd.DataFrame()
        
        # Simple cleanup: convert object columns to categorical
        for col in df.select_dtypes(include=['object']).columns:
            # Check for high cardinality before converting to category
            if df[col].nunique() < 50: # Arbitrary limit for category conversion
                df[col] = df[col].astype('category')
            
        return df
    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        return pd.DataFrame()


# --- Visualization Function ---
def create_interactive_bar_chart(data, x_col, color_col, agg_col, agg_func):
    """
    Creates a Plotly interactive bar chart based on user selections.
    """
    if data.empty or x_col is None or color_col is None or agg_col is None:
        return

    # Ensure the aggregation column is numeric and drop invalid rows
    data[agg_col] = pd.to_numeric(data[agg_col], errors='coerce')
    data.dropna(subset=[x_col, color_col, agg_col], inplace=True)
    
    if data.empty:
        st.warning(f"No valid data remaining after filtering for non-null values in {x_col}, {color_col}, and {agg_col}.")
        return

    try:
        # 1. Group and aggregate the data
        agg_name = "Count" if agg_func == pd.Series.count else "Sum"
        
        summary_df = data.groupby([x_col, color_col]).agg(
            {agg_col: agg_func}
        ).reset_index()
        
        summary_df.columns = [x_col, color_col, 'Aggregated Value']
        
        # 2. Create the Plotly bar chart
        fig = px.bar(
            summary_df,
            x=x_col,
            y='Aggregated Value',
            color=color_col,
            title=f"{agg_name} of {agg_col} by {x_col} (Split by {color_col})",
            labels={'Aggregated Value': f'{agg_name} of {agg_col}', x_col: x_col},
            barmode='stack', # Defaulting to stacked as per the original requirement
        )

        # 3. Customization
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=f'{agg_name} of {agg_col}',
            font=dict(family="Public Sans", size=12),
            title_font_size=16,
            legend_title_text=color_col
        )
        
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during chart generation: {e}")


# --- Main App Logic ---
def main():
    st.title("📊 Dynamic Interactive Stacked Bar Chart Dashboard")
    st.markdown("Upload your data file and use the sidebar to configure the visualization.")

    # 1. File Uploader in Sidebar
    st.sidebar.header("📥 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    
    df = pd.DataFrame()
    if uploaded_file is not None:
        df = load_data(uploaded_file)

    if df.empty:
        st.warning("Please upload a file to proceed.")
        return

    # Identify column types
    categorical_cols = df.select_dtypes(include=['category', 'object', 'int64', 'bool']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude columns that are likely IDs or highly unique categories
    if numeric_cols:
        # Simple check for ID-like columns (high cardinality but not float)
        potential_id_cols = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.9]
        categorical_cols = [col for col in categorical_cols if col not in potential_id_cols]

    if not categorical_cols:
        st.warning("The dataset contains no suitable columns for categorical grouping (X-Axis or Split).")
        return


    # 2. Sidebar for User Selections
    st.sidebar.header("🛠️ Chart Controls")
    
    # Select X-axis Category (main grouping)
    selected_x_col = st.sidebar.selectbox(
        "Select **X-Axis Category** (Main Bar Grouping)",
        options=categorical_cols,
        index=0 if categorical_cols else None
    )
    
    # Select Splitting/Color Category
    color_options = [col for col in categorical_cols if col != selected_x_col]
    selected_color_col = st.sidebar.selectbox(
        "Select **Splitting Category** (Bar Stacking/Color)",
        options=color_options,
        index=0 if color_options else None
    )

    # Select Aggregation Column (the values being aggregated)
    selected_agg_col = st.sidebar.selectbox(
        "Select **Value Column** (The numeric column being aggregated)",
        options=numeric_cols,
        index=0 if numeric_cols else None
    )

    # Select Aggregation Type
    aggregation_options = {
        'Sum of Value': np.sum, 
        'Count of Records': pd.Series.count
    }
    
    # Determine available aggregation options
    if selected_agg_col is None:
        current_agg_options = {'Count of Records': pd.Series.count}
        selected_agg_label = 'Count of Records'
        st.sidebar.info("Only 'Count of Records' is available as no numeric column was selected.")
    else:
        current_agg_options = aggregation_options
        selected_agg_label = st.sidebar.selectbox(
            "Select **Aggregation Type**",
            options=list(current_agg_options.keys()),
            index=0
        )
    
    selected_agg_func = current_agg_options[selected_agg_label]

    # 3. Generate and Display Chart (only if essential columns are selected)
    if all([selected_x_col, selected_color_col]) and (selected_agg_col or selected_agg_label == 'Count of Records'):
        
        # If 'Count of Records' is selected, ensure a column is used for aggregation (it doesn't matter which, as long as it's not null)
        final_agg_col = selected_agg_col if selected_agg_col else selected_x_col 

        with st.container():
            create_interactive_bar_chart(
                df.copy(), 
                selected_x_col, 
                selected_color_col, 
                final_agg_col, 
                selected_agg_func
            )
    else:
        st.info("Please select the required X-Axis and Splitting categories, and a Value column (unless only counting records).")

    # 4. Display Raw Data (Optional)
    if st.checkbox('Show Raw Data Table'):
        st.subheader('Raw Data')
        st.dataframe(df)

if __name__ == '__main__':
    main()
