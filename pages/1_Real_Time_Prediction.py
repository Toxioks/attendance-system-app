# Import and load required libraries.
import streamlit as st 
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

# Insert brand logo to sidebar, resizing logo to fit column.
st.sidebar.image("logo.png", use_column_width=True)

# Create a header with blue text to 'ATTEND' and include a rainbow divider.
st.header('FACE:blue[ATTEND]',divider='rainbow')

# Display a subheader for page title.
st.subheader('Real-Time Clock-in')

# Display a spinner message while retrieving data from the database.
with st.spinner('Retrieving data from database...'):
    # Retrieve face recognition data from Redis database using a custom key.
    redis_face_db = face_rec.retrieve_data(name='academy:register')
    # Display a success message once data is retrieved.
    st.success("Data successfully retrieved from database",icon="✅")

# Set the wait time for the real-time prediction loop.
waitTime = 30 # time in seconds.
# Record the current time to manage the prediction loop timing.
setTime = time.time()
# Instantiate the RealTimePred class for real-time face prediction.
realtimepred = face_rec.RealTimePred() 

# Define the callback function for processing video frames (Real-Time Prediction).
def video_frame_callback(frame):
    global setTime # Access the global setTime variable.
    # Convert the incoming video frame to a 3D numpy array.
    img = frame.to_ndarray(format="bgr24") 
    # Perform face prediction on the image array using the retrieved database and predefined threshold.
    pred_img = realtimepred.face_prediction(img,redis_face_db,
                                        'facial_features',['Name','Role'],thresh=0.5)
    
    # Calculate the time elapsed since the last database save operation.
    timenow = time.time()
    difftime = timenow - setTime
    # If the wait time has passed, save logs to Redis and reset the timer.
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time() # reset time        
        print('Save Data to redis database')
    
    # Return the processed video frame
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# Start the real-time video streamer with the specified key and callback function.
webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback,
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] 
    }
)