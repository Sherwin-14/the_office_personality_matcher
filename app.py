import numpy as np
import pandas as pd
import streamlit as st
import pickle
import os
from PIL import Image

# Load the data
df = pickle.load(open('data/data.pkl', 'rb'))

# Get unique characters
unique_characters = df['speaker'].unique().tolist()

# Streamlit app title
st.title("The Office US Personality Matcher")

# Character selection
selected_character = st.selectbox("Select a character", unique_characters)

# Load the images of each character
character_images = {}
for character in unique_characters:
    for file in os.listdir("images"):
        if file.startswith(character) and file.endswith((".jpg", ".jpeg", ".png")):
            image_path = f"images/{file}"
            character_images[character] = Image.open(image_path)
            break
    if character not in character_images:
        character_images[character] = None

# Function to calculate Euclidean distance
def calculate_euclidean_distance(vector1, vector2):
    x1, y1 = vector1['x'], vector1['y']
    x2, y2 = vector2['x'], vector2['y']
    return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

# Function to find the closest character
def find_closest_character(selected_character, character_vectors):
    if selected_character not in character_vectors:
        return "Character not found"
    
    closest_character = None
    min_distance = float('inf')
    for character, vector in character_vectors.items():
        if character != selected_character:
            distance = calculate_euclidean_distance(character_vectors[selected_character], vector)
            if distance < min_distance:
                min_distance = distance
                closest_character = character
    return closest_character

# Assuming you have a dictionary of character vectors
character_vectors = df.set_index('speaker')[['x', 'y']].to_dict('index')

# Find the recommended character
recommended_character = find_closest_character(selected_character, character_vectors)
recommended_character_image = character_images.get(recommended_character, None)

# Display the selected and recommended characters
cols = st.columns(2)

with cols[0]:
    st.header(selected_character)
    if character_images[selected_character]:
        st.image(character_images[selected_character])
    else:
        st.write("Image not available")

with cols[1]:
    st.header(recommended_character)
    if recommended_character_image:
        st.image(recommended_character_image)
    else:
        st.write("Image not available")




