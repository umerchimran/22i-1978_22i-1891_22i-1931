import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import librosa
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Function to read metadata files
def read_metadata_files(metadata_dir):
    albums = pd.read_csv(os.path.join(metadata_dir, 'raw_albums.csv'))
    artists = pd.read_csv(os.path.join(metadata_dir, 'raw_artists.csv'))
    genres = pd.read_csv(os.path.join(metadata_dir, 'raw_genres.csv'))
    tracks = pd.read_csv(os.path.join(metadata_dir, 'raw_tracks.csv'), dtype={'track_id': str})
    return albums, artists, genres, tracks

# Function to preprocess metadata
def preprocess_metadata(albums, artists, genres, tracks):
    # Merge metadata DataFrames
    metadata = pd.merge(tracks, albums, on='album_id', how='left')
    metadata = pd.merge(metadata, artists, on='artist_id', how='left')

    # Create a list to store track-genre mappings
    track_genre_mappings = []

    # Extract genre information from 'track_genres' column
    for index, row in tracks.iterrows():
        track_id = row['track_id']
        track_genres = eval(row['track_genres']) if isinstance(row['track_genres'], str) else []
        for genre in track_genres:
            genre_id = genre['genre_id']
            track_genre_mappings.append({'track_id': track_id, 'genre_id': genre_id})

    # Convert list of mappings to DataFrame
    track_genre_mapping = pd.DataFrame(track_genre_mappings)

    # Merge track-genre mapping DataFrame with genres DataFrame
    metadata = pd.merge(metadata, track_genre_mapping, on='track_id', how='left')
    
    # Convert 'genre_id' column to the same data type as in genres DataFrame
    metadata['genre_id'] = metadata['genre_id'].fillna(-1).astype(genres['genre_id'].dtype)

    # Merge with genres DataFrame
    metadata = pd.merge(metadata, genres, on='genre_id', how='left')

    # Handle missing values in string columns
    string_columns = metadata.select_dtypes(include=['object']).columns
    metadata[string_columns] = metadata[string_columns].fillna('Unknown')

    return metadata

# Function to extract audio features
def extract_audio_features(audio_dir):
    audio_features = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".mp3"):
                audio_file = os.path.join(root, file)
                try:
                    features = extract_features(audio_file)
                    if features is not None:
                        audio_features.append(features)
                except Exception as e:
                    print(f"Warning: Failed to process audio file '{audio_file}': {e}")
    return np.array(audio_features)

# Function to extract features from audio file
def extract_features(audio_file):
    try:
        # Try to load audio file using librosa
        y, sr = librosa.load(audio_file, sr=None)
    except Exception as e:
        # If loading fails, print a warning and return None
        raise RuntimeError(f"Failed to load audio file '{audio_file}': {e}")
    
    # Proceed with feature extraction if loading is successful
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

    # Aggregate statistics over each feature
    features = []
    for feature in [mfcc, spectral_centroid, zero_crossing_rate]:
        features.append(np.mean(feature))
        features.append(np.std(feature))
        features.append(np.median(feature))
    return features

# Function to preprocess audio features
def preprocess_audio_features(features):
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=5)
    reduced_features = pca.fit_transform(scaled_features)
    
    return reduced_features

# Function to upload data to MongoDB
def upload_to_mongodb(metadata, audio_features):
    # Connect to MongoDB
    client = MongoClient('localhost', 27017)
    db = client['music_data']
    metadata_collection = db['metadata']
    audio_collection = db['audio_features']

    # Convert DataFrame to dictionary for MongoDB insertion
    metadata_dict = metadata.to_dict(orient='records')

    # Insert metadata into MongoDB
    metadata_collection.insert_many(metadata_dict)

    # Insert audio features into MongoDB
    for i, feature in tqdm(enumerate(audio_features), total=len(audio_features), desc="Uploading audio features to MongoDB"):
        document = {
            "features": feature.tolist(),
            "track_id": metadata.iloc[i]['track_id'] if 'track_id' in metadata.columns else f"Unknown_track_{i+1}"
        }
        audio_collection.insert_one(document)

def main():
    # Step 1: Read metadata files
    metadata_dir = '/home/umar/BigDataProj/fma_metadata'
    albums, artists, genres, tracks = read_metadata_files(metadata_dir)
    
    # Step 2: Preprocess metadata
    metadata = preprocess_metadata(albums, artists, genres, tracks)

    # Step 3: Extract audio features
    audio_dir = '/home/umar/BigDataProj/fma_large/fma_large'
    audio_features = extract_audio_features(audio_dir)

    # Step 4: Preprocess audio features
    audio_features = preprocess_audio_features(audio_features)

    # Step 5: Upload data to MongoDB
    upload_to_mongodb(metadata, audio_features)

if __name__ == "__main__":
    main()

