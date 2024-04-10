# Import and load required libraries.
import streamlit as st 
import pandas as pd
from Home import face_rec

# Insert brand logo to sidebar, resizing logo to fit column.
st.sidebar.image("logo.png", use_column_width=True)

# Create a header with blue text to 'ATTEND' and include a rainbow divider.
st.header('FACE:blue[ATTEND]',divider='rainbow')

# Display a subheader for page title.
st.subheader('Admin panel')

# Initialize the registration form object from the face_rec module.
registration_form = face_rec.RegistrationForm()

# Function to load logs data from Redis and display in the report.
name = 'attendance:logs'
def load_logs(name,end=-1):
    # Retrieve all data from the specified Redis list.
    logs_list = face_rec.r.lrange(name,start=0,end=end) # extract all data from the redis database
    return logs_list

# Create tabs in the Streamlit app for different reports and actions.
tab1, tab2, tab3, tab4 = st.tabs(['Real-Time Prediciton Logs','User Clock-in data','Registered user report','Delete Registered User'])

# Tab for real-time prediction logs.
with tab1:
    # Button to refresh and retrieve the latest logs.
    if st.button('Refresh Logs'):
        with st.spinner('Retrieving data from database...'):
            # Display success message and the retrieved logs.
            st.success("Data successfully retrieved from database",icon="✅")
            st.write(load_logs(name=name))
# Tab for user clock-in data report.
with tab2:
    st.subheader('Attendance Report')
    logs_list = load_logs(name=name)
    # Step 1: Convert the logs from bytes to string for readability.
    convert_byte_to_string = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_byte_to_string, logs_list))
    
    # Step 2: Split each log string by '@' to create a nested list.
    split_string = lambda x: x.split('@')
    logs_nested_list = list(map(split_string, logs_list_string))

    # Convert the nested list into a pandas DataFrame for analysis.
    logs_df = pd.DataFrame(logs_nested_list, columns= ['Name','Role','Timestamp'])

    # Step 3: Convert the 'Timestamp' column to datetime for time-based analysis.
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date
    
    # Step 4: Group the DataFrame by date, name, and role to calculate in-time and out-time.
    report_df = logs_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'), # Earliest timestamp as in-time.
        Out_time = pd.NamedAgg('Timestamp','max') # Latest timestamp as out-time.
    ).reset_index()

    # Convert 'In_time' and 'Out_time' to datetime for calculation.
    report_df['In_time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out_time'] = pd.to_datetime(report_df['Out_time']) # Calculate the duration between in-time and out-time.
    report_df['Duration'] = report_df['Out_time'] - report_df['In_time'] # Display the final report DataFrame.

    # Step 5: Mark each person as present or absent based on their attendance duration.
    all_dates = report_df['Date'].unique() # Get all unique dates from the report.
    name_role = report_df[['Name','Role']].drop_duplicates().values.tolist() # Get unique name-role pairs.
    
    date_name_rol_zip = []
    # Create a list of all possible date-name-role combinations.
    for dt in all_dates:
        for name, role in name_role:
            date_name_rol_zip.append([dt,name,role])
            # Convert the list to a DataFrame.
    date_name_rol_zip_df = pd.DataFrame(date_name_rol_zip,columns=['Date','Name','Role'])
    
    # Join the new DataFrame with the original report DataFrame to fill in attendance details.
    date_name_rol_zip_df = pd.merge(date_name_rol_zip_df,report_df,how='left',on=['Date','Name','Role'])

    # Calculate the duration in hours from the 'Duration' column.
    date_name_rol_zip_df['Duration_seconds'] = date_name_rol_zip_df['Duration'].dt.seconds
    date_name_rol_zip_df['Duration_hours'] = date_name_rol_zip_df['Duration_seconds'] / (60*60)

    # Define a function to mark the attendance status based on the duration.
    def status_marker(x):
        if pd.Series(x).isnull().all():
            return 'Absent' # Mark as absent if no duration is recorded.
        elif x >= 0 and x < 1:
            return 'Absent (Less than 1 hour)' # Mark as absent if present for less than 1 hour.
        elif x >= 1 and x < 4:
            return 'Half day (Less than 4 hours)' # Mark as half-day if present for less than 4 hours.
        elif x >= 4 and x < 6:  
            return 'Half day' # Mark as half-day if present for 4-6 hours.
        elif x >= 6: 
            return 'Present' # Mark as present if present for 6 or more hours.

    # Apply the status marker function to each row to determine the attendance status.
    date_name_rol_zip_df['Status'] = date_name_rol_zip_df['Duration_hours'].apply(status_marker) 

    # Display the DataFrame with the attendance status in the Streamlit app.
    st.dataframe(date_name_rol_zip_df)

# Tab for registered user report.
with tab3:
    if st.button('Refresh Data'):
        # Button to refresh and retrieve the latest registered user data.
        with st.spinner('Retriving Data from Redis DB ...'):    
             # Retrieve the registered user data from the Redis database.
            redis_face_db = face_rec.retrieve_data(name='academy:register')
            # Display the retrieved data as a DataFrame.
            st.dataframe(redis_face_db)

# Tab for deleting a registered user.
with tab4:
    st.subheader('Deletion Form')
    # Input fields for the name and role of the user to be deleted.
    key_person_name = st.text_input(label='Key_Name', placeholder='First & Last Name')
    key_role = st.selectbox(label='Key_Select Role',options=('Student',
                                                 'Teacher'))
    # Button to initiate the deletion process.
    if st.button('Delete'):
        # Attempt to remove the user data from the Redis database.
        return_key_val = registration_form.remove_data_in_redis_db(key_person_name,key_role)
        if return_key_val == True:
            # Display a success message if the deletion is successful.
            st.success(f"{key_person_name} deleted sucessfully")
        elif return_key_val == 'name_false':
            # Display an error message if the deletion fails.
            st.error('Please enter the name: Name cannot be empty or spaces')
