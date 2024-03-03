import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Function to load JSON data from an UploadedFile object
def load_json_from_uploaded_file(uploaded_file):
    try:
        # Read the contents of the uploaded file
        json_data = uploaded_file.read().decode("utf-8")
        # Parse the JSON data
        return json.loads(json_data)
    except json.JSONDecodeError:
        st.error("Error: Failed to decode JSON data. Make sure the file is in valid JSON format.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading JSON data: {e}")
        return None

# Function to extract text data from JSON
def extract_text_from_json(json_data):
    try:
        texts = [entry.get('data', {}).get('text', '') for entry in json_data]
        if not texts:
            st.warning("No text data found in the JSON file.")
        return texts
    except Exception as e:
        st.error(f"An error occurred while extracting text data from JSON: {e}")
        return None

# Function to vectorize text data
def vectorize_text(texts):
    try:
        if not texts:
            st.warning("No text data to vectorize.")
            return None

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()
    except Exception as e:
        st.error(f"An error occurred during text vectorization: {e}")
        return None, None

# Function to save vectorized data to a file
def save_vectorized_data(vectorized_data, filename):
    try:
        np.save(filename, vectorized_data)
        st.success(f"Vectorized data saved to {filename}")
    except Exception as e:
        st.error(f"An error occurred while saving vectorized data: {e}")

# Streamlit UI
st.title("Text Vectorization")

# Upload JSON file
json_file = st.file_uploader("Upload JSON file", type=["json"])

# Input field for folder path
output_folder_path = st.text_input("Enter the output folder path:", value="/home/gray/Downloads/Devai/Vector_data")

if json_file is not None:
    try:
        # Load JSON data from the uploaded file
        json_data = load_json_from_uploaded_file(json_file)

        if json_data is not None:
            # Extract text data from JSON
            texts = extract_text_from_json(json_data)

            if texts is not None:
                # Perform vectorization
                vectorized_data, _ = vectorize_text(texts)

                if vectorized_data is not None:
                    # Ensure the output folder path exists
                    try:
                        os.makedirs(output_folder_path, exist_ok=True)
                    except Exception as e:
                        st.error(f"An error occurred while creating the output folder: {e}")
                        st.stop()

                    # Find the next available filename for saving vectorized data
                    i = 1
                    while os.path.exists(os.path.join(output_folder_path, f"vec_{i}.npy")):
                        i += 1
                    filename = os.path.join(output_folder_path, f"vec_{i}.npy")

                    # Save vectorized data to file
                    save_vectorized_data(vectorized_data, filename)
    except Exception as e:
        st.error(f"An error occurred: {e}")
