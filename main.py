import streamlit as st
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# --- Configuration and Helper Functions ---

st.set_page_config(page_title="Disaster Response Dashboard", layout="wide")

# Caching helps to avoid re-running expensive functions on every interaction
@st.cache_resource
def get_llm():
    """Initializes and returns the Language Model."""
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
        model="deepseek/deepseek-r1-0528-qwen3-8b:free",
        temperature=0.1 # Lower temperature for more predictable JSON output
    )

# Use geopy for address-to-coordinate conversion, with caching
@st.cache_data
def geocode_locations(df):
    """
    Adds 'latitude' and 'longitude' columns to the DataFrame by geocoding the 'location' column.
    """
    if 'location' not in df.columns or df['location'].isnull().all():
        df['latitude'] = None
        df['longitude'] = None
        return df

    geolocator = Nominatim(user_agent="disaster_dashboard_app")
    # RateLimiter prevents sending too many requests too quickly
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, error_wait_seconds=10)
    
    df_location = df[['location']].copy().dropna().drop_duplicates('location')
    
    # Create a progress bar for user feedback
    progress_bar = st.progress(0, text="Geocoding locations...")
    latitudes, longitudes = [], []

    for i, location_str in enumerate(df_location['location']):
        try:
            location_geo = geocode(location_str)
            if location_geo:
                latitudes.append(location_geo.latitude)
                longitudes.append(location_geo.longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
        except Exception:
            latitudes.append(None)
            longitudes.append(None)
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(df_location), text=f"Geocoding: {location_str}")
        
    progress_bar.empty() # Clear the progress bar when done
    
    df_location['latitude'] = latitudes
    df_location['longitude'] = longitudes
    
    # Merge the geocoded data back into the original dataframe
    df = df.merge(df_location, on='location', how='left')
    return df


# --- Main Application UI ---

st.title("🚨 Disaster Situation Dashboard")
st.markdown("Upload a text file containing disaster reports to automatically extract data and visualize the situation.")

# Define the master prompt for extracting a list of JSON objects
extraction_prompt_template = """
You are a highly efficient data extraction system. Your task is to analyze the entire document provided below and identify every reported incident or request for help.

For each distinct incident you find, create a JSON object with the following schema. If a piece of information for a field is not available, you MUST use `null`.

JSON Schema for each incident:
- `victimName`: The name of the victim or person reporting.
- `location`: The specific location of the incident (e.g., address, city, landmark).
- `needs`: A brief description of what is required.
- `timestamp`: The date and time the information was reported.
- `degreeOfNeed`: Must be one of: "critical", "urgent", "moderate", "mild".
- `typeOfNeed`: Must be one of: "food", "water", "shelter", "medical", "rescue", "other".
- `source`: The origin of the information (e.g., "Twitter", "Facebook", "News Report").

Your final output must be a single, valid JSON array (a list of JSON objects), with one object for each incident found. Do not include any other text, explanations, or markdown formatting. If the document contains no incidents, return an empty array `[]`.

Document to analyze:
---
{document_text}
---
"""

uploaded_file = st.file_uploader("Upload a `.txt` file with disaster reports", type=["txt"])

if uploaded_file:
    # Read the content of the uploaded file
    document_text = uploaded_file.read().decode("utf-8")
    
    if st.button("Process Document and Build Dashboard"):
        with st.spinner("Analyzing document with AI... This may take a moment."):
            try:
                llm = get_llm()
                prompt = PromptTemplate(template=extraction_prompt_template, input_variables=["document_text"])
                chain = LLMChain(llm=llm, prompt=prompt)
                
                # Run the extraction chain
                llm_output = chain.run(document_text)
                
                # The output should be a string containing a JSON array. Let's parse it.
                # Find the start and end of the JSON array to handle potential LLM chatter
                start = llm_output.find('[')
                end = llm_output.rfind(']') + 1
                json_str = llm_output[start:end]
                
                extracted_data = json.loads(json_str)

                if not extracted_data:
                    st.warning("No incident data could be extracted from the document.")
                    st.stop()
                
                # Convert the extracted data into a pandas DataFrame
                df = pd.DataFrame(extracted_data)
                
                # Geocode locations to get coordinates for the map
                df_geocoded = geocode_locations(df.copy())
                
                # Store the processed data in Streamlit's session state to use across reruns
                st.session_state.df_geocoded = df_geocoded

            except json.JSONDecodeError:
                st.error("AI model failed to return a valid JSON array. It might be due to model limitations or the document's content.")
                st.code(llm_output, language="text")
                st.stop()
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.stop()

# --- Display Dashboard if data is available in session state ---
if 'df_geocoded' in st.session_state:
    df_geocoded = st.session_state.df_geocoded
    st.success(f"Successfully extracted {len(df_geocoded)} incidents!")

    # --- Sidebar for Filtering ---
    st.sidebar.header("Filter Dashboard")
    
    # Filter by Degree of Need
    need_degrees = df_geocoded['degreeOfNeed'].dropna().unique()
    selected_degrees = st.sidebar.multiselect(
        "Filter by Urgency", options=need_degrees, default=need_degrees
    )
    
    # Filter by Type of Need
    need_types = df_geocoded['typeOfNeed'].dropna().unique()
    selected_types = st.sidebar.multiselect(
        "Filter by Need Type", options=need_types, default=need_types
    )

    # Apply filters
    filtered_df = df_geocoded[
        (df_geocoded['degreeOfNeed'].isin(selected_degrees)) &
        (df_geocoded['typeOfNeed'].isin(selected_types))
    ]

    # --- Main Dashboard Layout ---
    st.header("Dashboard Summary")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reports", len(filtered_df))
    critical_count = filtered_df[filtered_df['degreeOfNeed'] == 'critical'].shape[0]
    col2.metric("Critical Needs", critical_count)
    medical_count = filtered_df[filtered_df['typeOfNeed'] == 'medical'].shape[0]
    col3.metric("Medical Requests", medical_count)

    st.markdown("---")
    
    # Visualizations
    col1_viz, col2_viz = st.columns(2)
    
    with col1_viz:
        st.subheader("Needs by Type")
        need_counts = filtered_df['typeOfNeed'].value_counts()
        st.bar_chart(need_counts)

    with col2_viz:
        st.subheader("Incidents by Urgency")
        urgency_counts = filtered_df['degreeOfNeed'].value_counts()
        st.bar_chart(urgency_counts)
        
    # Map Visualization
    st.subheader("Incident Locations")
    map_data = filtered_df.dropna(subset=['latitude', 'longitude'])
    if not map_data.empty:
        st.map(map_data)
    else:
        st.info("No geocoded locations to display on the map for the current selection.")

    # Detailed Data View
    with st.expander("Show Detailed Report Data"):
        st.dataframe(filtered_df)