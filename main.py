import streamlit as st
# Spotify API:
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
# In built python module to compute large mathematical data:
import numpy as np
# Importing the module needed to normalize numbers between 0 and 1--> Everyhing will be between 0 and 1  
from sklearn.preprocessing import MinMaxScaler
# Importing module for cosine similarity to compare vectors based on scalar quantity and not magnitude
from sklearn.metrics.pairwise import cosine_similarity
# To convert to vector database:
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Tuple
# numerical assistanrt ti h
import pandas as pd
import time

class AIRecommender:
    def __init__(self, sp_client):
        """Initialize AI components and Spotify client."""
        self.sp = sp_client
        self.scaler = MinMaxScaler()
        
        # Initialize AI models
        # essentially check whether everything is being initialized correctly
        try:
            # Emotion analysis model. Using prior Hugging Face model to check
            self.emotion_analyzer = pipeline('text-classification', 
                                          model='j-hartmann/emotion-english-distilroberta-base', 
                                          top_k=2)
            
            # Text embedding model for mood analysis: 
            self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize feature mapping. These features are being used from Spotify API results for each song
            # More robust data so that song predictions are accurate
            self.AUDIO_FEATURES = [
                'danceability', 'energy', 'valence', 'tempo',
                'acousticness', 'instrumentalness', 'speechiness', 'liveness'
            ]
            
            # Create mood embeddings
            self.mood_embeddings = self._create_mood_embeddings()
            
        except Exception as e:
            st.error(f"Error initializing AI models: {str(e)}")
            raise # Highlighting and bringing the error to stage

    def _create_mood_embeddings(self) -> Dict[str, np.ndarray]:
        #Creates a dictionary mapping moods to descriptive words. Words similar to our base categories.
        mood_descriptions = {
            "Happy": "joyful excited cheerful upbeat bright positive energetic",
            "Sad": "melancholic depressed gloomy down blue unhappy lonely",
            "Energetic": "dynamic powerful strong intense active vigorous",
            "Calm": "peaceful relaxed tranquil serene gentle quiet soothing",
            "Focused": "concentrated determined productive motivated mindful"
        }
        # Converting each mood description/ category into a vector in the vector database.
        # Then, it returns the dictionary where the mood is the key and vector is the value
        return {mood: self.text_embedder.encode(desc) 
                for mood, desc in mood_descriptions.items()}

    def analyze_mood(self, text_input: str) -> Tuple[str, Dict[str, float]]:
        """Analyze mood using multiple AI models."""
        try:
            # Uses the Hugging Face algorithm and models to iterate over each mood or feeling to give it a score
            # Get emotion probabilities
            emotions = self.emotion_analyzer(text_input)
            # creating a dictionary of moods and feelings mapped to their relevant similiarity scores
            emotion_scores = {item['label']: item['score'] 
                            for item in emotions[0]}
            
            # Converting the user input into a vector so that we can compare
            input_embedding = self.text_embedder.encode(text_input)
            
            # Calculate similarity with mood categories
            mood_scores = {}
            # Comparing the scores of the computer to the user to find similiarity using vector cosines
            for mood, mood_embedding in self.mood_embeddings.items():
            #The cosine_similarity() function returns a 2D array, we access the similarity score to get the scalar value.
             # The calculated similarity score is  stored in the mood_scores dictionary, with the mood as the key and the similarity score as the value.
             #Embeds the user's input with the embeddings of various predefined moods, calculating how similar they are using cosine similarity.
             #Cosine similarity is a measure of similarity between two non-zero vectors. It's calculated as the cosine of the angle between the vectors, resulting in a value between -1 and 1. A value closer to 1 indicates higher similarity.
             # because range of cosine baby! 
             # #After this loop completes, mood_scores will contain similarity scores for each mood, allowing you to determine which mood(s) are most similar to the input.
                similarity = cosine_similarity(
                    input_embedding.reshape(1, -1),
                    mood_embedding.reshape(1, -1)
                )[0][0]
                # creates a dictionary with the mode as the key and the similarity score as the value
                mood_scores[mood] = similarity
            
            # Combine emotion and mood scores. Returns the overall dominant mood based on all the parameters
            # Uses this dominant mood (highest similarity) to find the genre of music to compare with
            dominant_mood = max(mood_scores.items(), key=lambda x: x[1])[0]
            
            return dominant_mood, mood_scores
            
        except Exception as e:
            st.error(f"Error in mood analysis: {str(e)}")
            return "Happy", {"Happy": 1.0}

    def extract_track_features(self, track_id: str) -> np.ndarray:
        # The arrow in the function tells us the type of data that is being returned
        # Returning an array but we are using numpy because we are dealing with complex data
        """Extract and process audio features using AI."""
        try:
            # Using Spotify API to get audio features
            features = self.sp.audio_features(track_id)[0]
            if not features:
                return None
            
            # Creating vectors for each audio feature and adding it to the numpy array
            feature_vector = np.array([features[feat] for feat in self.AUDIO_FEATURES])
            
            # Normalizing the vector features --> reshaping it between 1 and -1 (since we are using cosine similarity)
            normalized_features = self.scaler.fit_transform(feature_vector.reshape(1, -1))
            
            return normalized_features
            
        except Exception as e:
            st.error(f"Error extracting features: {str(e)}")
            return None

    def calculate_similarity_score(
        self, 
        track_features: np.ndarray, #Converting to numpy array types
        seed_features: List[np.ndarray], #Converting to a list numpy array types
        mood_scores: Dict[str, float] # Converting to a dictionary with the mood as key and similarity score as value
    ) -> float: #return type is going to be a float
        """Calculate similarity score using multiple factors."""
        # Finding the cosine similarity between track features and seed tracks
        similarities = [
            cosine_similarity(track_features, seed_feat.reshape(1, -1))[0][0]
            for seed_feat in seed_features
        ]
        # Setting the category of audio similarity to be the average of all the similarity scores calculated for it.
        audio_similarity = np.mean(similarities)
        
        # Finds the score for the moods of each song 
        mood_compatibility = sum(
            score * self._calculate_mood_feature_match(track_features, mood)
            for mood, score in mood_scores.items()
        )
        
        # Weighted combination so that we have more importance to moods such as happy, etc as opposed to audio features like valence
        final_score = (audio_similarity * 0.4) + (mood_compatibility * 0.6)
        return final_score

    def _calculate_mood_feature_match(
        self, 
        features: np.ndarray, 
        mood: str
    ) -> float:
        """Calculate how well track features match a mood."""
        # Assigning default weightage to the different features from Spotify
        feature_weights = {
            "Happy": {'valence': 0.5, 'energy': 0.4, 'danceability': 0.2},
            "Sad": {'valence': 0.4, 'energy': 0.3, 'acousticness': 0.4},
            "Energetic": {'energy': 0.5, 'tempo': 0.1, 'danceability': 0.4},
            "Calm": {'energy': 0.4, 'acousticness': 0.3, 'instrumentalness': 0.3},
            "Focused": {'valence': 0.3, 'energy': 0.3, 'instrumentalness': 0.4}
        }
        # Getting the weighted scores based on the different audio features
        weights = feature_weights.get(mood, feature_weights["Sad"])
        score = 0.0
        
        for feature, weight in weights.items():
            feature_idx = self.AUDIO_FEATURES.index(feature)
            score += weight * features[0, feature_idx]
         # Iterating over the rates of each mood and creating a score for the user
        return score

    def get_recommendations(
        self, 
        artist_name: str,
        mood_input: str,
        limit: int = 10
    ) -> Tuple[List[Dict], List[float], Dict[str, float]]:
        """Get AI-powered music recommendations."""
        try:
            # Analyze mood using AI
            detected_mood, mood_scores = self.analyze_mood(mood_input)
            
            # Search for artist and return no artist if nothing is found
            results = self.sp.search(q=artist_name, type='artist', limit=1)
            if not results['artists']['items']:
                st.error("Artist not found")
                return [], [], mood_scores
            # Setting the baseline for the artist
            artist = results['artists']['items'][0]
            
            # Get artist's top tracks
            top_tracks = self.sp.artist_top_tracks(artist['id'])['tracks']
            
            # Extract features from seed tracks
            seed_features = []
            for track in top_tracks[:3]:
                features = self.extract_track_features(track['id'])
                if features is not None:
                    seed_features.append(features)

            if not seed_features:
                return [], [], mood_scores

            # Get initial recommendations
            recommendations = self.sp.recommendations(
                seed_tracks=[track['id'] for track in top_tracks[:2]],
                limit=limit * 2
            )

            # Score recommendations using AI
            scored_recommendations = []
            for track in recommendations['tracks']:
                # Getting the features of each track that is recommended
                features = self.extract_track_features(track['id'])
                if features is not None:
                    # Using cosine similarity to find the scores of each feature for the song
                    score = self.calculate_similarity_score(
                        features,
                        seed_features,
                        mood_scores
                    )
                    # Appending the name of each song along with its similarity score 
                    scored_recommendations.append((track, score))

            # Sorting by the best similarity score 
            scored_recommendations.sort(key=lambda x: x[1], reverse=True)
            # Arranging it based on the first recommendation all the way till the last recommendation based on the number of songs the user wants
            recommendations = [rec[0] for rec in scored_recommendations[:limit]]
            scores = [rec[1] for rec in scored_recommendations[:limit]]

            return recommendations, scores, mood_scores

        except Exception as e:
            st.error(f"Error in recommendation process: {str(e)}")
            return [], [], {}

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Music Recommender",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize Spotify client
    try:
        sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id="d1abdd8d2c1d48058ae2128e03f5e1d2",
                client_secret="9faed433292a475ebdcd5bc0be37b758"
            )
        )
        recommender = AIRecommender(sp)
    except Exception as e:
        st.error(f"Error connecting to Spotify API: {str(e)}")
        st.stop()

    # Main Header
    st.title("🎵 AI Music Mood Matcher")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("✨ How it Works")
        st.info("""
        This smart system uses multiple AI models to:
        
        🧠 **Mood Analysis**
        - Processes your mood description
        - Understands emotional context
        
        🎵 **Music Analysis**
        - Analyzes audio features
        - Matches musical patterns
        
        🤖 **AI Matching**
        - Uses neural networks
        - Finds perfect song matches
        """)
        
        st.divider()
        st.header("📊 Quick Stats")
        st.metric("AI Models", "3")
        st.metric("Music Features", "8")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🎯 Get Recommendations", "ℹ️ About"])
    
    with tab1:
        st.header("What would you like to listen to?")
        
        # Input section - avoid nested columns
        artist_name = st.text_input(
            "🎤 Artist Name",
            placeholder="e.g., Taylor Swift",
            help="Enter your favorite artist's name"
        )
        
        mood_input = st.text_area(
            "😊 Your Current Mood",
            placeholder="e.g., I'm feeling energetic and ready to dance",
            help="Describe how you're feeling right now"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            num_recommendations = st.slider(
                "Number of recommendations",
                min_value=5,
                max_value=20,
                value=10
            )
            
            min_popularity = st.slider(
                "Minimum song popularity",
                min_value=0,
                max_value=100,
                value=50,
                help="0 = least popular, 100 = most popular"
            )
        
        # Get recommendations button
        if st.button("🎯 Get AI Recommendations", type="primary", use_container_width=True):
            if not artist_name or not mood_input:
                st.error("🎵 Please enter both artist name and mood description")
                return
            
            with st.spinner("🎵 AI is analyzing your mood and finding perfect matches..."):
                recommendations, scores, mood_scores = recommender.get_recommendations(
                    artist_name,
                    mood_input,
                    num_recommendations
                )
            
            if recommendations:
                # Display mood analysis
                st.header("🎭 Your Mood Analysis")
                st.divider()
                
                # Mood scores
                mood_cols = st.columns(len(mood_scores))
                for col, (mood, score) in zip(mood_cols, mood_scores.items()):
                    with col:
                        st.metric(
                            label=mood,
                            value=f"{score:.1%}",
                            delta=None,
                        )
                
                st.divider()
                st.success(f"🎉 Found {len(recommendations)} perfect matches!")
                st.divider()
                
                # Display recommendations
                for i in range(0, len(recommendations), 3):
                    cols = st.columns(3)
                        # Album image and track info side by side
                    for j, col in enumerate(cols):
                        if i + j < len(recommendations):
                            track = recommendations[i + j]
                            #score = scores[i + j]

                            with col:
                                if track['album']['images']:
                                    st.image(
                                        track['album']['images'][0]['url'],
                                        use_column_width=True
                                    )
                                st.markdown(f"**{track['name']}**")
                                st.write(f"by {track['artists'][0]['name']}")
                                st.write(f"Album**: {track['album']['name']}")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(
                                        f"[![Play on Spotify]"
                                        f"(https://img.shields.io/badge/Spotify-1ED760?&style=for-the-badge&logo=spotify&logoColor=white)]"
                                        f"({track['external_urls']['spotify']})"
                                    )
                                if track['preview_url']:
                                    with col2:
                                        st.audio(track['preview_url'])
                            
    with tab2:
        st.header("About this App")
        st.write("""
        This AI-powered music recommender uses state-of-the-art machine learning to:
        
        - Analyze your current mood through natural language processing
        - Extract and analyze audio features from songs
        - Match songs to your mood using neural networks
        - Provide personalized recommendations based on your favorite artists
        """)
        
        st.divider()
        
        st.subheader("The system considers:")
        st.write("""
        - Mood compatibility
        - Musical features
        - Artist similarity
        - Song popularity
        """)
        
        st.divider()
        
        # App statistics
        st.subheader("📊 System Features")
        st.metric("AI Models", "3", "Used for analysis")
        st.metric("Audio Features", "8", "Analyzed per song")
        st.metric("Mood Categories", "5", "For matching")

if __name__ == "__main__":
    main()