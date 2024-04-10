# Import & load Numpy, Pandas, and openCV library.
import numpy as np
import pandas as pd
import cv2

# Import and load redis library
import redis

# Import & load insight face library
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Import & load time library.
import time
from datetime import datetime

# Import & load os library.
import os


# Connect to Redis Client / Database.
hostname = 'redis-19202.c100.us-east-1-4.ec2.cloud.redislabs.com'
portnumber = 19202
password = '6Ou9GxlORn01Ev0PXQupVCWwUMn1aUce'

# Initialise Redis client with given hostname, port, and password.
r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

# Function to Retrieve Data from database.
def retrieve_data(name):
    # Get all hash values from Redis based on the name key.
    retrieve_dict= r.hgetall(name)
    # Convert hash values to Pandas Series.
    retrieve_series = pd.Series(retrieve_dict)
    # Decode byte data and convert to numpy array of type float32.
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    # Decode the index bytes to string.
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    # Update the index of the series.
    retrieve_series.index = index
    # Convert series to DataFrame and reset index.
    retrieve_df =  retrieve_series.to_frame().reset_index()
    # Set DataFrame column names.
    retrieve_df.columns = ['name_role','facial_features']
    # Split 'name_role' into 'Name' and 'Role' and expand into separate columns.
    retrieve_df[['Name','Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    # Return the final DataFrame with selected columns
    return retrieve_df[['Name','Role','facial_features']]


# Initialize and configure face analysis application.
faceapp = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CPUExecutionProvider'])
# Prepare the face analysis model with given context ID and detection settings.
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# Function ML Search Algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector,
                        name_role=['Name', 'Role'], thresh=0.5):
    """
    Implements a cosine similarity-based search algorithm.
    It compares a test vector against a feature column in a dataframe and
    returns the name and role of the entry with the highest similarity above a given threshold.
    """
    # Make a copy of the dataframe to avoid changes to the original data.
    dataframe = dataframe.copy()
    # Extract the feature data from the dataframe and convert it into a numpy array.
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    # Calculate the cosine similarity between the feature data and the test vector.
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    # Flatten the resulting array to a 1D array for easier handling.
    similar_arr = np.array(similar).flatten()
    # Add the similarity scores as a new column in the dataframe.
    dataframe['cosine'] = similar_arr

    # Filter the dataframe to only include rows where the similarity score is above the threshold.
    data_filter = dataframe.query(f'cosine >= {thresh}')
    # If there are any rows that meet the criteria.
    if len(data_filter) > 0:
        # Reset the index of the filtered dataframe.
        data_filter.reset_index(drop=True, inplace=True)
        # Find the index of the row with the highest similarity score.
        argmax = data_filter['cosine'].argmax()
        # Retrieve the name and role of the entry with the highest similarity score.
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        # If no entry meets the threshold, return 'Unknown' for both name and role.
        person_name = 'Unknown'
        person_role = 'Unknown'
        
    # Return the key/name and role of the person with the highest similarity score.
    return person_name, person_role


# This class is designed to handle real-time prediction and logging of data.

class RealTimePred:
    def __init__(self):
        # Initialize a dictionary to store logs with keys: name, role, and current_time.
        self.logs = dict(name=[], role=[], current_time=[])
        
    def reset_dict(self):
        # Reset the logs dictionary to empty lists for each key.
        self.logs = dict(name=[], role=[], current_time=[])
        
    def saveLogs_redis(self):
        # This method saves the logs to a Redis database.
        
        # Step 1: Convert the logs dictionary into a pandas DataFrame.
        dataframe = pd.DataFrame(self.logs)        
        # Step 2: Remove duplicate entries based on the 'name' column to ensure distinct names.
        dataframe.drop_duplicates('name', inplace=True) 
        # Step 3: Prepare and encode the data for insertion into the Redis database.
        
        # Extract lists of names, roles, and times from the dataframe.
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        
        # Encode the data into a format suitable for Redis.
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            # Exclude entries with the name 'Unknown'.
            if name != 'Unknown':
                # Concatenate the name, role, and time into a single string.
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
                
        # If there is encoded data, push it to the Redis list 'attendance:logs'.
        if len(encoded_data) > 0:
            r.lpush('attendance:logs', *encoded_data)
        
        # Reset the logs dictionary after saving to Redis.
        self.reset_dict()
        
    # Function Face prediction
    def face_prediction(self,test_image, dataframe,feature_column,
                            name_role=['Name','Role'],thresh=0.5):
        # Step 1: Obtain the current time as a string.
        current_time = str(datetime.now())
        
        # Step 2: Retrieve face recognition results using the insight face application.
        results = faceapp.get(test_image)
        # Create a copy of the test image.
        test_copy = test_image.copy()
        
        # Step 3: Iterate over each result to process and identify faces..
        for res in results:
            # Extract the bounding box coordinates and convert them to integers.
            x1, y1, x2, y2 = res['bbox'].astype(int)
            # Retrieve the face embedding vector.
            embeddings = res['embedding']
            # Use the machine learning search algorithm to find the closest match in the database.
            person_name, person_role = ml_search_algorithm(dataframe,
                                                        feature_column,
                                                        test_vector=embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
            # Set the rectangle color to red for 'Unknown' or green for recognized individuals.
            if person_name == 'Unknown':
                color =(0,0,255) # bgr
            else:
                color = (0,255,0)

            # Draw a rectangle around the face and add the person's name and current time as text.
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)

            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            
            # Step 4: Save the identified person's name, role, and the current time in the logs dictionary.
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
            
        # Return the annotated image.
        return test_copy


#### Registration Form
class RegistrationForm:
    def __init__(self):
        # Initialize a sample counter to zero.
        self.sample = 0
    def reset(self):
        # Reset the sample counter back to zero.
        self.sample = 0
        
    def get_embedding(self,frame):
        # Retrieve results from the insightface model with a maximum of one result.
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            # Increment the sample counter for each face detected.
            self.sample += 1
            # Extract bounding box coordinates.
            x1, y1, x2, y2 = res['bbox'].astype(int)
            # Draw a rectangle around the face.
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            # Display the number of samples processed on the frame.
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            
            # Extract and store facial embeddings.
            embeddings = res['embedding']
        
        # Return the frame with drawn rectangles and the embeddings.
        return frame, embeddings
    
    def save_data_in_redis_db(self,name,role):
        # Validate the provided name.
        if name is not None:
            if name.strip() != '':
                # Create a unique key using name and role.
                key = f'{name}@{role}'
            else:
                # Return an error if the name is invalid.
                return 'name_false'
        else:
            return 'name_false'
        
        # Check if the file 'face_embedding.txt' exists in the current directory.
        if 'face_embedding.txt' not in os.listdir():
            # Return an error if the file does not exist.
            return 'file_false'
        
        
        # Load embeddings from 'face_embedding.txt' as a flat array.
        x_array = np.loadtxt('face_embedding.txt',dtype=np.float32) # flatten array            
        
        # Reshape the array to the proper format based on the number of samples.
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)       
        
        # Calculate the mean of the embeddings.
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        # Convert the mean embeddings to bytes for storage.
        x_mean_bytes = x_mean.tobytes()
        
        # Save the mean embeddings into the Redis database under the 'academy:register' hash.
        r.hset(name='academy:register',key=key,value=x_mean_bytes)
        
        # Remove the 'face_embedding.txt' file after processing.
        os.remove('face_embedding.txt')
        # Reset the sample counter.
        self.reset()
        
        # Return the Redis connection object (assuming 'r' is a Redis connection object).
        return True
    
    def remove_data_in_redis_db(self,key,role):
                # Set the base name for the Redis hash where data is stored.
                name='academy:register'
                # Validate the provided key.
                if key is not None:
                    # Ensure the key is not empty after stripping whitespace.
                    if key.strip() != '':
                        # Format the key to include the role, creating a unique identifier.
                        key_to_del = str(f'{key}@{role}')
                    else:
                        # Return an error if the key is invalid.
                        return 'name_false'
                else:
                    # Return an error if the key is None.
                    return 'name_false'
            
                # Delete the specified key-value pair from the Redis hash.
                r.hdel(name,key_to_del)
                
                # Reset the form to its initial state after deletion.
                self.reset()
                
                # Return True to indicate successful deletion.
                return True