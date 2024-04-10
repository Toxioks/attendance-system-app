# Import Streamlit library
import streamlit as st

# Set configuration for Streamlit dynamic page.
st.set_page_config(page_title='Attendance System',layout='wide')

# Insert brand logo to sidebar, resizing logo to fit column.
st.sidebar.image("logo.png", use_column_width=True)

# Create a header with blue text to 'ATTEND' and include a rainbow divider.
st.header('FACE:blue[ATTEND]',divider='rainbow')
# Display a subheader for page title.
st.subheader('Web based facial recognition online register')

# Display a spinner while database creates a connection and loads required models.
with st.spinner("Loading Data, Models and Connecting to Redis database..."):
    # Import face_rec.py 
    import face_rec
    
# Display success messages once the model is loaded and the database connection is established.
st.success('Loaded Data, Models and Connected to Redis database successfully',icon="✅")
