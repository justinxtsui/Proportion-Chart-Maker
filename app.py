import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Configuration and Setup ---
st.set_page_config(
    page_title="Dynamic Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the custom colors from the original Matplotlib script
# NOTE: The categories in the 'Splitting Category' must match these keys ('Pipeline', 'Primary') 
# for the custom colors and text colors to work correctly.
CUSTOM_COLORS = {
    'Pipeline': '#EDD9E4',  # Light color for Pipeline
    'Primary': '#6F2A58'    # Dark color for Primary
}

# --- Data Loading Function ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded CSV or Excel file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file (.csv, .xlsx, .xls).")
            return pd.DataFrame()
        
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 50:
                df[col] = df[col].astype('category')
            
        return df
    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        return pd.DataFrame()


# --- Visualization Function (Corrected and Customized for Proportional Stacked Bars) ---
def create_styled_proportional_bar_chart(data, x_col, color_col, agg_col, agg_func):
    """
    Creates a Plotly proportional stacked bar chart, mimicking the original design
    and ensuring the proportional logic is correctly applied (sum to 100%).
    """
    if data.empty or x_col is None or color_col is None or agg_col is None:
        return

    # 1. Prepare Data for Proportional Plot
    data[agg_col] = pd.to_numeric(data[agg_col], errors='coerce')
    data.dropna(subset=[x_col, color_col, agg_col], inplace=True)
    
    if data.empty:
        st.warning(f"No valid data remaining after filtering for non-null values.")
        return

    try:
        # Group and aggregate
        summary_df = data.groupby([x_col, color_col]).agg({agg_col: agg_func}).reset_index()
        summary_df.columns = [x_col, color_col, 'Aggregated Value']
        
        # Calculate Proportions (Essential for the requested look: always 0-100%)
        total_by_x = summary_df.groupby(x_col)['Aggregated Value'].transform('sum')
        summary_df['Proportion'] = (summary_df['Aggregated Value'] / total_by_x) * 100
        
        # 2. Create the Plotly Figure object
        fig = go.Figure()
        
        # Ensure a defined order for stacking and colors. Use keys from CUSTOM_COLORS if possible.
        ordered_categories = [cat for cat in CUSTOM_COLORS.keys() if cat in summary_df[color_col].unique()]
        # Add any other categories found in the data, just in case
        for cat in summary_df[color_col].unique():
            if cat not in ordered_categories:
                ordered_categories.append(cat)
                
        # Initialize the baseline for stacking
        current_bottom = pd.Series([0.0] * summary_df[x_col].nunique(), 
                                    index=summary_df[x_col].unique()).sort_index()

        # --- Add Traces and Data Labels ---
        for category in ordered_categories:
            cat_data = summary_df[summary_df[color_col] == category].copy()
            
            # Reindex to ensure all X categories are present, which introduces NaNs
            cat_data = cat_data.set_index(x_col).reindex(current_bottom.index).reset_index()
            cat_data['Bottom'] = current_bottom
            
            # --- FIX: Only fill numerical columns to avoid TypeError ---
            numerical_cols_to_fill = ['Aggregated Value', 'Proportion', 'Bottom']
            cat_data[numerical_cols_to_fill] = cat_data[numerical_cols_to_fill].fillna(0)
            # --- END FIX ---
            
            cat_data['Top'] = cat_data['Bottom'] + cat_data['Proportion']
            
            # Determine color and text color based on category name
            marker_color = CUSTOM_COLORS.get(category, '#A9A9A9') # Default to Dark Gray
            text_color = 'black' if category == 'Pipeline' else '#D3D3D3' # Light gray
            
            # Add Bar Trace
            fig.add_trace(go.Bar(
                x=cat_data[x_col],
                y=cat_data['Proportion'],
                name=f'{category} scaleups' if category in CUSTOM_COLORS else category, # Match original legend
                marker_color=marker_color,
                base=cat_data['Bottom'],
                customdata=cat_data[['Proportion']],
                hovertemplate=f"{x_col}: %{{x}}<br>{category}: %{{customdata[0]:.1f}}%<extra></extra>"
            ))
            
            # Add Data Labels (Percentage inside bars)
            for i, row in cat_data.iterrows():
                proportion = row['Proportion']
                if proportion > 5: # Only label if proportion is large enough
                    y_position = row['Bottom'] + (proportion / 2)
                    fig.add_annotation(
                        x=row[x_col],
                        y=y_position,
                        text=f"{proportion:.1f}%",
                        showarrow=False,
                        font=dict(
                            family="Public Sans",
                            size=12,
                            color=text_color,
                            weight='bold'
                        ),
                        xanchor='center',
                        yanchor='middle'
                    )
            
            # Update current bottom for the next stack
            current_bottom = cat_data['Top'].set_index(cat_data[x_col])
        
        # 3. Apply Styling
        
        fig.update_layout(
            barmode='stack',
            xaxis_title=None,
            yaxis_title=None,
            yaxis=dict(
                range=[0, 100], 
                showgrid=False,
                showticklabels=False,
                fixedrange=True, # Prevent user from zooming on Y-axis
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=12, family="Public Sans"),
                showline=False 
            ),
            title={
                'text': f'Proportion of {agg_col} by {x_col}', # Dynamic title update
                'font': {'size': 14, 'weight': 'bold', 'family': 'Public Sans'},
                'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
            },
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05, 
                font=dict(size=18, family="Public Sans"),
                traceorder="normal",
                bgcolor='rgba(0,0,0,0)',
            ),
            margin=dict(l=20, r=200, t=60, b=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)
        
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during chart generation: {e}")
        st.exception(e)


# --- Main App Logic ---
def main():
    st.title("💰 IUK Grant Style Analysis Dashboard")
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
    
    if selected_agg_col is None:
        current_agg_options = {'Count of Records': pd.Series.count}
        selected_agg_label = 'Count of Records'
        st.sidebar.info("Only 'Count of Records' is available.")
    else:
        current_agg_options = aggregation_options
        selected_agg_label = st.sidebar.selectbox(
            "Select **Aggregation Type**",
            options=list(current_agg_options.keys()),
            index=0
        )
    
    selected_agg_func = current_agg_options[selected_agg_label]
    
    # Final check and chart generation
    if all([selected_x_col, selected_color_col]) and (selected_agg_col or selected_agg_label == 'Count of Records'):
        
        final_agg_col = selected_agg_col if selected_agg_col else selected_x_col 

        st.markdown("---")
        st.info(
            "The chart below is a **proportional stacked bar chart (0-100%)**. "
            "The segments in each bar represent the percentage contribution of the "
            "Splitting Category to the total value/count of the X-Axis Category."
        )
        
        with st.container():
            create_styled_proportional_bar_chart(
                df.copy(), 
                selected_x_col, 
                selected_color_col, 
                final_agg_col, 
                selected_agg_func
            )
    else:
        st.info("Please select the required X-Axis, Splitting category, and a Value column to generate the chart.")

    # 4. Display Raw Data (Optional)
    if st.checkbox('Show Raw Data Table'):
        st.subheader('Raw Data')
        st.dataframe(df)

if __name__ == '__main__':
    main()
