# Import and load required libraries.
import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

# Insert brand logo to sidebar, resizing logo to fit column.
st.sidebar.image("logo.png", use_column_width=True)

# Create a header with blue text to 'ATTEND' and include a rainbow divider.
st.header('FACE:blue[ATTEND]',divider='rainbow')

# Display a subheader for page title.
st.subheader('User Registration')

## init the Registration form object
registration_form = face_rec.RegistrationForm()

# Step-1: Collect person's name and role via form inputs.
person_name = st.text_input(label='Name',placeholder='First & Last Name') # Text input for name.
role = st.selectbox(label='Select your Role',options=('Student',
                                                      'Teacher')) # Dropdown for selecting role.


# Step-2: Define a callback function to collect facial embeddings via video.
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') # Convert frame to 3D array in BGR format.
    reg_img, embedding = registration_form.get_embedding(img)
    # Get registration image and embedding.
    # If an embedding is obtained, save it to a local file.
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f: # Open file in append binary mode.
            np.savetxt(f,embedding) # Save the embedding as text.
    
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')  # Return the processed video frame.

# Initialize the video streamer with the callback function. 
webrtc_streamer(key='registration',video_frame_callback=video_callback_func,
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# Step-3: Save the collected data into a Redis database upon button click
if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,role)
    # Attempt to save data in Redis.
    # Handle the response based on the return value.
    if return_val == True:
        st.success(f"{person_name} registered sucessfully") # Success message
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces') # Error for empty name.
        
    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute again.') # Error for missing file.
        