import streamlit as st 
import pandas as pd
from Home import face_rec

# Insert brand logo into sidebar
st.sidebar.image("logo.png", use_column_width=True)

st.header('FACE:blue[ATTEND]',divider='rainbow')
st.subheader('Admin panel')

registration_form = face_rec.RegistrationForm()

# Retrive logs data and show in Report.py
# extract data from redis list
name = 'attendance:logs'
def load_logs(name,end=-1):
    logs_list = face_rec.r.lrange(name,start=0,end=end) # extract all data from the redis database
    return logs_list

# tabs to show the info
tab1, tab2, tab3, tab4 = st.tabs(['Real-Time Prediciton Logs','User Clock-in data','Registered user report','Delete Registered User'])

with tab1:
    if st.button('Refresh Logs'):
        with st.spinner('Retrieving data from database...'):
            st.success("Data successfully retrieved from database",icon="✅")
            st.write(load_logs(name=name))

with tab2:
    st.subheader('Attendance Report')
    logs_list = load_logs(name=name)
    # step 1: Convert the logs in list of bytes to string
    convert_byte_to_string = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_byte_to_string, logs_list))
    
    # step 2: split string by @ and create nested list
    split_string = lambda x: x.split('@')
    logs_nested_list = list(map(split_string, logs_list_string))

    # convert nested list into dataframe
    logs_df = pd.DataFrame(logs_nested_list, columns= ['Name','Role','Timestamp'])

    # step 3: Time based analysis report
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date
    
    # step 4: Calculate in-time and out-time
    # In time: Detected first time in particular date. Out time: Detected last time in particular date.
    report_df = logs_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'), # In-time
        Out_time = pd.NamedAgg('Timestamp','max') # Out-time
    ).reset_index()

    report_df['In_time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out_time'] = pd.to_datetime(report_df['Out_time'])
    report_df['Duration'] = report_df['Out_time'] - report_df['In_time']

    # Step 5: mark as present or Absent
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name','Role']].drop_duplicates().values.tolist()
    
    date_name_rol_zip = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_rol_zip.append([dt,name,role])
    date_name_rol_zip_df = pd.DataFrame(date_name_rol_zip,columns=['Date','Name','Role'])
    
    # Join with report_df

    date_name_rol_zip_df = pd.merge(date_name_rol_zip_df,report_df,how='left',on=['Date','Name','Role'])

    # Duration (hours)
    date_name_rol_zip_df['Duration_seconds'] = date_name_rol_zip_df['Duration'].dt.seconds
    date_name_rol_zip_df['Duration_hours'] = date_name_rol_zip_df['Duration_seconds'] / (60*60)

    def status_marker(x):
        if pd.Series(x).isnull().all():
            return 'Absent'
        elif x >= 0 and x < 1:
            return 'Absent (Less than 1 hour)'
        elif x >= 1 and x < 4:
            return 'Half day (Less than 4 hours)'
        elif x >= 4 and x < 6: 
            return 'Half day'
        elif x >= 6: 
            return 'Present'

    date_name_rol_zip_df['Status'] = date_name_rol_zip_df['Duration_hours'].apply(status_marker) 

    st.dataframe(date_name_rol_zip_df)

with tab3:
    if st.button('Refresh Data'):
        # Retrieve the data from Redis Database
        with st.spinner('Retriving Data from Redis DB ...'):    
            redis_face_db = face_rec.retrieve_data(name='academy:register')
            st.dataframe(redis_face_db)

with tab4:
    st.subheader('Deletion Form')
    key_person_name = st.text_input(label='Key_Name', placeholder='First & Last Name')
    key_role = st.selectbox(label='Key_Select Role',options=('Student',
                                                 'Teacher'))
    if st.button('Delete'):
        return_key_val = registration_form.remove_data_in_redis_db(key_person_name,key_role)
        if return_key_val == True:
            st.success(f"{key_person_name} deleted sucessfully")
        elif return_key_val == 'name_false':
            st.error('Please enter the name: Name cannot be empty or spaces')
