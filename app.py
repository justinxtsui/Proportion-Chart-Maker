import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
import base64

# --- Configuration and Setup ---
st.set_page_config(
    page_title="Proportional Count Chart Maker",
    layout="centered", 
    initial_sidebar_state="auto"
)

# Define the custom colors from the original Matplotlib script
CUSTOM_COLORS = {
    'Pipeline': '#EDD9E4',  # Light purple
    'Primary': '#6F2A58'    # Dark purple
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


# --- Visualization Function (Matches Matplotlib Design & Static) ---
def create_styled_proportional_bar_chart(data, x_col, color_col):
    """
    Creates a Plotly proportional stacked bar chart, rigorously matching the Matplotlib design.
    """
    if data.empty or x_col is None or color_col is None:
        return None
    
    data.dropna(subset=[x_col, color_col], inplace=True)
    
    if data.empty:
        st.warning("No valid data remaining after filtering for non-null grouping values.")
        return None

    try:
        # 1. Prepare Data for Proportional Plot (Based on COUNT)
        summary_df = data.groupby([x_col, color_col]).size().reset_index(name='Aggregated Value')
        total_by_x = summary_df.groupby(x_col)['Aggregated Value'].transform('sum')
        summary_df['Proportion'] = (summary_df['Aggregated Value'] / total_by_x) * 100
        
        # 2. Create the Plotly Figure object
        fig = go.Figure()
        
        # --- Define categories in a fixed, required order ---
        # NOTE: This assumes the categories, when sorted, are plotted in this order, 
        # or we iterate through the colors directly.
        
        # Check if the required keys are present, and use them first
        required_categories = list(CUSTOM_COLORS.keys())
        
        ordered_categories = [cat for cat in required_categories if cat in summary_df[color_col].unique()]
        # Add any other categories for robustness, although they won't have custom colors
        for cat in summary_df[color_col].unique():
            if cat not in ordered_categories:
                ordered_categories.append(cat)

        current_bottom = pd.Series([0.0] * summary_df[x_col].nunique(), 
                                    index=summary_df[x_col].unique()).sort_index()

        # --- Add Traces and Data Labels ---
        for category in ordered_categories:
            cat_data = summary_df[summary_df[color_col] == category].copy()
            
            x_template = pd.DataFrame(index=current_bottom.index)
            cat_data = cat_data.set_index(x_col)
            
            cat_data = x_template.merge(cat_data, left_index=True, right_index=True, how='left')
            cat_data.reset_index(inplace=True)
            cat_data.rename(columns={'index': x_col}, inplace=True) 
            
            cat_data['Bottom'] = current_bottom.reset_index(drop=True)
            
            cat_data[['Aggregated Value', 'Proportion']] = cat_data[['Aggregated Value', 'Proportion']].fillna(0)
            
            cat_data['Top'] = cat_data['Bottom'] + cat_data['Proportion']
            
            # --- Styling Logic Matching Matplotlib ---
            category_str = str(category)
            
            # Use fixed colors/text colors for the two defined categories
            if category_str == 'Pipeline':
                marker_color = CUSTOM_COLORS['Pipeline'] # #EDD9E4
                text_color = 'black'
                plot_name = 'Pipeline scaleups'
            elif category_str == 'Primary':
                marker_color = CUSTOM_COLORS['Primary'] # #6F2A58
                text_color = '#D3D3D3' # Light gray
                plot_name = 'Primary scaleups'
            else:
                # Default for unexpected categories (should not happen for the original data)
                marker_color = '#A9A9A9'
                text_color = 'black'
                plot_name = category_str
            
            # Add Bar Trace
            fig.add_trace(go.Bar(
                x=cat_data[x_col],
                y=cat_data['Proportion'],
                name=plot_name,
                marker_color=marker_color,
                base=cat_data['Bottom'],
                width=0.6, 
                hoverinfo='skip' 
            ))

            # Add Data Labels (Percentage inside bars)
            for i, row in cat_data.iterrows():
                proportion = row['Proportion']
                if proportion > 5:
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
            
            current_bottom = cat_data.set_index(x_col)['Top']
        
        # 3. Apply Global Styling (Spines, Ticks, Legend, Title)
        
        fig.update_layout(
            barmode='stack',
            xaxis_title=None,
            yaxis_title=None,
            yaxis=dict(
                range=[0, 100], 
                showgrid=False,
                showticklabels=False,
                showline=False, 
                fixedrange=True,
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=12, family="Public Sans"),
                showline=False 
            ),
            title={
                'text': 'Proportion of Grant Amounts by Year', 
                'font': {'size': 14, 'weight': 'bold', 'family': 'Public Sans'},
                'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
                'pad': {'b': 20} 
            },
            # Legend styling: fixed labels and colors based on Matplotlib design
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05, 
                font=dict(size=18, family="Public Sans"),
                traceorder="normal",
                bgcolor='rgba(0,0,0,0)', 
                
                # --- Force custom legend structure ---
                # NOTE: This overrides the trace names if they don't match, 
                # but ensures the legend always looks correct.
                # However, since we define trace names above, let's trust the traces, 
                # but ensure markers are correct.
                
                # Forcing custom markers (circles) is not clean in Plotly layouts. 
                # We will rely on the trace colors and font size/position.
            ),
            margin=dict(l=20, r=200, t=60, b=20),
            plot_bgcolor='white', 
            paper_bgcolor='white',
            
            # Disable all interactive tools
            modebar_remove=['zoom', 'pan', 'select', 'lasso', 'autoscale', 'reset', 'toimage', 'hovercompare', 'togglehover']
        )

        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)
        
        return fig

    except Exception as e:
        st.error(f"An error occurred during chart generation: {e}")
        st.exception(e)
        return None

# --- Download SVG Function ---
def get_svg_download_link(fig, filename="chart.svg"):
    """Generates an HTML download link for the SVG file using base64 encoding."""
    buffer = BytesIO()
    try:
        fig.write_image(buffer, format="svg", width=1400, height=800) 
    except ValueError as e:
        st.error("Error generating SVG. Please ensure you have the 'kaleido' library installed: `pip install kaleido`")
        return None

    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    href = f'<a href="data:image/svg+xml;base64,{b64}" download="{filename}">Download SVG Chart</a>'
    return href


# --- Main App Logic ---
def main():
    st.title("📊 Proportional Stacked Bar Chart Tool (Static Replica)")
    st.markdown("This tool generates a chart with the exact design specifications for high-quality SVG export.")

    # 1. File Uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    
    df = pd.DataFrame()
    if uploaded_file is not None:
        df = load_data(uploaded_file)

    if df.empty:
        st.info("Please upload a file to proceed.")
        return

    # Identify column types
    categorical_cols = df.select_dtypes(include=['category', 'object', 'int64', 'bool']).columns.tolist()
    
    if not categorical_cols:
        st.warning("The dataset contains no suitable columns for categorical grouping.")
        return

    # 2. Controls 
    with st.container():
        st.header("Chart Configuration")
        col1, col2 = st.columns(2)
        
        # Select X-axis Category (main grouping)
        with col1:
            selected_x_col = st.selectbox(
                "1. Select **X-Axis Category** (Main Bar Grouping)",
                options=categorical_cols,
                index=0 if categorical_cols else None
            )
        
        # Select Splitting/Color Category
        with col2:
            color_options = [col for col in categorical_cols if col != selected_x_col]
            selected_color_col = st.selectbox(
                "2. Select **Splitting Category** (Bar Stacking/Color)",
                options=color_options,
                index=0 if color_options else None
            )
        
        st.caption(f"The chart will automatically aggregate by the **Count** of records.")

    # 3. Generate and Display Chart
    st.markdown("---")
    
    if all([selected_x_col, selected_color_col]):
        
        # Check if the required categories are present in the chosen column
        if not all(cat in df[selected_color_col].astype(str).unique() for cat in CUSTOM_COLORS.keys()):
            st.warning(
                f"The column **'{selected_color_col}'** does not contain both 'Pipeline' and 'Primary' values. "
                f"The colors and legend labels might be incorrect. Found values: {df[selected_color_col].astype(str).unique()}"
            )
        
        with st.spinner("Generating highly-styled chart..."):
            fig = create_styled_proportional_bar_chart(
                df.copy(), 
                selected_x_col, 
                selected_color_col
            )
        
        if fig:
            # Display the static Plotly chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Create and display the SVG download link
            download_link = get_svg_download_link(fig, filename=f"proportional_count_chart_{selected_x_col}.svg")
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)

    else:
        st.info("Please select the required X-Axis and Splitting category to generate the chart.")


if __name__ == '__main__':
    main()
