import streamlit as st

st.set_page_config(page_title='Attendance System',layout='wide')

# Insert brand logo into sidebar
st.sidebar.image("logo.png", use_column_width=True)

st.header('FACE:blue[ATTEND]',divider='rainbow')
st.subheader('Web based face recognition online register')

with st.spinner("Loading Models and Conneting to Redis db ..."):
    import face_rec
    
st.success('Model loaded sucesfully')
st.success('Redis db sucessfully connected')