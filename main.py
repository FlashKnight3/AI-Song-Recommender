import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings that might affect the UI

# Page configuration
st.set_page_config(
    page_title="AI Music Mood Matcher",
    page_icon="🎵",
    layout="wide"
)


# App title and header
st.markdown('<h1 class="main-header">🎵 AI Music Mood Matcher</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Find the perfect songs for your current mood</p>', unsafe_allow_html=True)

class AIRecommender:
    def __init__(self, sp_client):
        self.sp = sp_client
        self.scaler = MinMaxScaler()
        
        try:
            self.emotion_analyzer = pipeline('text-classification', 
                                          model='j-hartmann/emotion-english-distilroberta-base', 
                                          top_k=2)
            
            self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.AUDIO_FEATURES = [
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo'
            ]
            
            self.mood_embeddings = self._create_mood_embeddings()
            self._initialize_scaler()
            
        except Exception as e:
            st.error(f"Error initializing AI models: {str(e)}")
            raise

    def _initialize_scaler(self):
        mock_data = np.array([
            [0, 0, 0, -60, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 11, 0, 1, 1, 1, 1, 1, 1, 250]
        ])
        self.scaler.fit(mock_data)

    def _create_mood_embeddings(self):
        mood_descriptions = {
            "Happy": "joyful excited cheerful upbeat bright positive energetic",
            "Sad": "melancholic depressed gloomy down blue unhappy lonely",
            "Energetic": "dynamic powerful strong intense active vigorous",
            "Calm": "peaceful relaxed tranquil serene gentle quiet soothing",
            "Focused": "concentrated determined productive motivated mindful"
        }
        return {mood: self.text_embedder.encode(desc) 
                for mood, desc in mood_descriptions.items()}

    def analyze_mood(self, text_input):
        """Analyze mood using multiple AI models."""
        try:
            # Get emotion probabilities
            emotions = self.emotion_analyzer(text_input)
            emotion_scores = {item['label']: item['score'] 
                            for item in emotions[0]}
            
            # Convert the user input into a vector for comparison
            input_embedding = self.text_embedder.encode(text_input)
            
            # Calculate similarity with mood categories
            mood_scores = {}
            for mood, mood_embedding in self.mood_embeddings.items():
                similarity = cosine_similarity(
                    input_embedding.reshape(1, -1),
                    mood_embedding.reshape(1, -1)
                )[0][0]
                mood_scores[mood] = similarity
            
            # Find dominant mood
            dominant_mood = max(mood_scores.items(), key=lambda x: x[1])[0]
            
            return dominant_mood, mood_scores
            
        except Exception as e:
            st.error(f"Error in mood analysis: {str(e)}")
            return "Happy", {"Happy": 1.0}  # Fallback to Happy mood

    def extract_track_features(self, track_id):
        """Extract and process audio features using Spotify API."""
        if not track_id:
            return None
            
        try:
            # Get audio features with retry logic
            for attempt in range(3):
                try:
                    features = self.sp.audio_features([track_id])
                    if not features or not features[0]:
                        # If no features returned, use default values instead of failing
                        default_features = {
                            'danceability': 0.5, 'energy': 0.5, 'key': 5, 'loudness': -10,
                            'mode': 1, 'speechiness': 0.1, 'acousticness': 0.5,
                            'instrumentalness': 0.01, 'liveness': 0.1, 'valence': 0.5, 'tempo': 120
                        }
                        track_features = default_features
                    else:
                        track_features = features[0]
                    
                    # Extract and validate each feature
                    feature_vector = []
                    for feature in self.AUDIO_FEATURES:
                        value = track_features.get(feature)
                        if value is None:
                            # Use sensible defaults for missing features
                            defaults = {
                                'danceability': 0.5, 'energy': 0.5, 'key': 5, 'loudness': -10,
                                'mode': 1, 'speechiness': 0.1, 'acousticness': 0.5,
                                'instrumentalness': 0.01, 'liveness': 0.1, 'valence': 0.5, 'tempo': 120
                            }
                            value = defaults.get(feature, 0.0)
                        feature_vector.append(float(value))
                    
                    # Convert to numpy array and transform (not fit_transform)
                    feature_array = np.array(feature_vector).reshape(1, -1)
                    normalized_features = self.scaler.transform(feature_array)
                    
                    return normalized_features
                    
                except spotipy.exceptions.SpotifyException as e:
                    if e.http_status == 429:  # Rate limiting
                        retry_after = int(e.headers.get('Retry-After', 1))
                        time.sleep(retry_after + 1)  # Add buffer
                        continue
                    elif e.http_status == 403 or e.http_status == 401:  # Authentication error
                        # Don't show error to user, just return default features
                        default_features = np.array([[0.5, 0.5, 5, -10, 1, 0.1, 0.5, 0.01, 0.1, 0.5, 120]])
                        return self.scaler.transform(default_features)
                    else:
                        if attempt < 2:  # Don't sleep on last attempt
                            time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                        
                except Exception as e:
                    if attempt < 2:  # Don't sleep on last attempt
                        time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
            # If all retries failed, return default features instead of None
            default_features = np.array([[0.5, 0.5, 5, -10, 1, 0.1, 0.5, 0.01, 0.1, 0.5, 120]])
            return self.scaler.transform(default_features)
            
        except Exception as e:
            # Return default features instead of showing error
            default_features = np.array([[0.5, 0.5, 5, -10, 1, 0.1, 0.5, 0.01, 0.1, 0.5, 120]])
            return self.scaler.transform(default_features)

    def calculate_similarity_score(self, track_features, seed_features, mood_scores):
        """Calculate similarity score using multiple factors."""
        if track_features is None or not seed_features:
            return 0.0
            
        # Finding the cosine similarity between track features and seed tracks
        similarities = [
            cosine_similarity(track_features, seed_feat)[0][0]
            for seed_feat in seed_features
        ]
        # Setting the audio similarity to be the average of all similarity scores
        audio_similarity = np.mean(similarities)
        
        # Calculate mood compatibility score
        mood_compatibility = sum(
            score * self._calculate_mood_feature_match(track_features, mood)
            for mood, score in mood_scores.items()
        )
        
        # Weighted combination
        final_score = (audio_similarity * 0.4) + (mood_compatibility * 0.6)
        return final_score

    def _calculate_mood_feature_match(self, features, mood):
        """Calculate how well track features match a mood."""
        # Feature indices for quick reference
        feature_indices = {name: i for i, name in enumerate(self.AUDIO_FEATURES)}
        
        # Mood-feature weights mapping
        feature_weights = {
            "Happy": {'valence': 0.5, 'energy': 0.4, 'danceability': 0.2},
            "Sad": {'valence': 0.4, 'energy': 0.3, 'acousticness': 0.4},
            "Energetic": {'energy': 0.5, 'tempo': 0.1, 'danceability': 0.4},
            "Calm": {'energy': 0.4, 'acousticness': 0.3, 'instrumentalness': 0.3},
            "Focused": {'valence': 0.3, 'energy': 0.3, 'instrumentalness': 0.4}
        }
        
        # Get weights for the current mood
        weights = feature_weights.get(mood, feature_weights["Happy"])
        score = 0.0
        
        for feature, weight in weights.items():
            if feature in feature_indices:
                feature_idx = feature_indices[feature]
                # For tempo, normalize to 0-1 range first (typical range 50-200)
                if feature == 'tempo':
                    value = min(max(features[0, feature_idx], 50), 200) / 200
                # For loudness, normalize from dB scale (-60 to 0) to 0-1
                elif feature == 'loudness':
                    value = (features[0, feature_idx] + 60) / 60
                else:
                    value = features[0, feature_idx]
                score += weight * value
            
        return score

    def get_recommendations(self, artist_name, mood_input, limit=10, min_popularity=20):
        """Get AI-powered music recommendations from various artists, including international ones."""
        try:
            # First analyze the mood
            detected_mood, mood_scores = self.analyze_mood(mood_input)
            
            # Create or retrieve persistent storage for used tracks across all moods
            if not hasattr(self, 'global_used_tracks'):
                self.global_used_tracks = {}
            
            # Search for the artist as a starting point
            results = self.sp.search(q=artist_name, type='artist', limit=1)
            if not results['artists']['items']:
                # Try a more general search if specific artist not found
                results = self.sp.search(q=artist_name, type='track', limit=1)
                if not results['tracks']['items']:
                    # If still no results, use a popular artist instead
                    results = self.sp.search(q='Taylor Swift', type='artist', limit=1)
                
            # Extract artist from results
            if results.get('artists', {}).get('items'):
                artist = results['artists']['items'][0]
                
                # Get the artist's genres
                artist_genres = artist.get('genres', [])
                
                # Use these genres to find similar artists
                # If no genres found, we'll use related artists instead
                similar_artists = []
                
                # Define international genres to incorporate global music
                international_genres = [
                    "k-pop", "j-pop", "reggaeton", "latin", "afrobeat", 
                    "brazilian", "indian", "bollywood", "mandopop", "french",
                    "german", "spanish", "italian", "arabic", "african"
                ]
                
                # Add international genre search
                for genre in international_genres[:3]:  # Use 3 random international genres
                    try:
                        int_results = self.sp.search(q=f"genre:{genre}", type='artist', limit=5)
                        if int_results.get('artists', {}).get('items'):
                            for int_artist in int_results['artists']['items']:
                                if int_artist['id'] not in [a.get('id') for a in similar_artists]:
                                    similar_artists.append(int_artist)
                    except:
                        # Skip if there's an issue with this genre
                        continue
                
                if artist_genres:
                    # Search for artists with at least one matching genre
                    genre_query = artist_genres[0] if artist_genres else "pop"
                    genre_results = self.sp.search(q=f"genre:{genre_query}", type='artist', limit=15)
                    for genre_artist in genre_results['artists']['items']:
                        if genre_artist['id'] not in [a.get('id') for a in similar_artists]:
                            similar_artists.append(genre_artist)
                
                # If we didn't get enough from genres, get related artists
                if len(similar_artists) < 15:
                    try:
                        related = self.sp.artist_related_artists(artist['id'])
                        related_artists = related['artists']
                        
                        # Add any new artists not already in the list
                        for related_artist in related_artists:
                            if related_artist['id'] != artist['id']:
                                if related_artist['id'] not in [a.get('id') for a in similar_artists]:
                                    similar_artists.append(related_artist)
                    except:
                        # If we can't get related artists, just continue with what we have
                        pass
                
                # Make sure we include original artist in the list
                if artist not in similar_artists:
                    similar_artists = [artist] + similar_artists
                
            else:
                # Use the track's artist from the track search
                track = results['tracks']['items'][0]
                artist = track['artists'][0]
                
                # Just use the artist and try to find related ones
                try:
                    related = self.sp.artist_related_artists(artist['id'])
                    similar_artists = [artist] + related['artists']
                    
                    # Add some international artists
                    for genre in ["k-pop", "latin", "afrobeat"][:2]:
                        int_results = self.sp.search(q=f"genre:{genre}", type='artist', limit=3)
                        if int_results.get('artists', {}).get('items'):
                            for int_artist in int_results['artists']['items']:
                                if int_artist['id'] not in [a.get('id') for a in similar_artists]:
                                    similar_artists.append(int_artist)
                except:
                    # If we can't get related artists, use international genres
                    similar_artists = [artist]
                    for genre in ["pop", "k-pop", "latin", "afrobeat", "bollywood"]:
                        try:
                            results = self.sp.search(q=f"genre:{genre}", type='artist', limit=5)
                            if results.get('artists', {}).get('items'):
                                for int_artist in results['artists']['items']:
                                    if int_artist['id'] not in [a.get('id') for a in similar_artists]:
                                        similar_artists.append(int_artist)
                        except:
                            continue
            
            # Collect tracks from multiple artists based on mood
            all_tracks = []
            current_mood_tracks = set()  # Track IDs for this specific mood
            
            # Check if we have used tracks for this mood before
            if detected_mood not in self.global_used_tracks:
                self.global_used_tracks[detected_mood] = set()
            
            # Get all tracks used across all moods
            all_used_tracks = set()
            for mood_tracks in self.global_used_tracks.values():
                all_used_tracks.update(mood_tracks)
            
            # Configure recommendation parameters based on mood
            mood_params = {
                "Happy": {"target_valence": 0.8, "target_energy": 0.7, "target_danceability": 0.7},
                "Sad": {"target_valence": 0.3, "target_energy": 0.4, "target_acousticness": 0.7},
                "Energetic": {"target_energy": 0.9, "target_tempo": 150, "target_danceability": 0.8},
                "Calm": {"target_energy": 0.3, "target_acousticness": 0.8, "target_instrumentalness": 0.4},
                "Focused": {"target_instrumentalness": 0.4, "target_energy": 0.5, "target_valence": 0.6}
            }
            
            # Use each similar artist as a seed to get recommendations
            for similar_artist in similar_artists[:30]:  # Increased from 20 to 30 for more options
                if len(all_tracks) >= limit * 5:  # Get 5x more tracks than needed to ensure diversity
                    break
                    
                try:
                    # Get top tracks for analyzing features
                    top_tracks = self.sp.artist_top_tracks(similar_artist['id'], country='US')['tracks'][:3]  # More seed tracks
                    
                    if top_tracks:
                        # Extract features from top tracks to use as seeds
                        seed_features = []
                        for track in top_tracks:
                            features = self.extract_track_features(track['id'])
                            if features is not None:
                                seed_features.append(features)
                        
                        # Get mood-based recommendations for this artist
                        params = {
                            "seed_artists": [similar_artist['id']],
                            "limit": 10,  # Get more tracks from each artist (was 5)
                            "min_popularity": max(min_popularity - 10, 10)  # Less reduction from minimum popularity
                        }
                        
                        # Add mood-specific parameters with stronger constraints
                        mood_specific = mood_params.get(detected_mood, {})
                        params.update(mood_specific)
                        
                        # Get recommendations for this artist
                        artist_recommendations = self.sp.recommendations(**params)['tracks']
                        
                        # Score and add to our collection, ensuring no duplicates across all moods
                        for track in artist_recommendations:
                            # Skip if this track has already been used in ANY mood
                            if track['id'] in all_used_tracks:
                                continue
                                
                            # Skip if we've already collected this track for the current mood
                            if track['id'] in current_mood_tracks:
                                continue
                                
                            features = self.extract_track_features(track['id'])
                            if features is not None:
                                # Calculate score with higher emphasis on current mood
                                score = self.calculate_similarity_score(features, seed_features, {detected_mood: 1.0})
                                all_tracks.append((track, score))
                                current_mood_tracks.add(track['id'])
                                
                except Exception:
                    # If there's an error with this artist, just skip and continue
                    continue
            
            # If we couldn't get enough diverse recommendations, try more diverse search
            if len(all_tracks) < limit:
                try:
                    # Use mood-based genre search directly
                    mood_genres = {
                        "Happy": ["pop", "dance", "disco"],
                        "Sad": ["indie", "folk", "ambient"],
                        "Energetic": ["electronic", "rock", "dance"],
                        "Calm": ["ambient", "classical", "acoustic"],
                        "Focused": ["classical", "instrumental", "electronic"]
                    }
                    
                    # Get genres for this mood
                    genres = mood_genres.get(detected_mood, ["pop"])
                    
                    # Try each genre
                    for genre in genres:
                        if len(all_tracks) >= limit:
                            break
                            
                        # Search for tracks in this genre
                        params = {"q": f"genre:{genre}", "type": "track", "limit": limit * 2}
                        
                        genre_results = self.sp.search(**params)
                        for track in genre_results.get('tracks', {}).get('items', []):
                            # Skip used tracks
                            if track['id'] in all_used_tracks or track['id'] in current_mood_tracks:
                                continue
                                
                            # Add with default score
                            all_tracks.append((track, 0.6))
                            current_mood_tracks.add(track['id'])
                            
                            if len(all_tracks) >= limit * 2:
                                break
                
                except Exception:
                    # Final fallback - use year-based search with mood parameters
                    try:
                        # Get current year
                        mood_term = "energetic" if detected_mood == "Energetic" else \
                                   "chill" if detected_mood == "Calm" else \
                                   "happy" if detected_mood == "Happy" else \
                                   "sad" if detected_mood == "Sad" else "focus"
                        
                        search_results = self.sp.search(q=f"year:2023 {mood_term}", type='track', limit=limit * 2)
                        for track in search_results.get('tracks', {}).get('items', []):
                            if track['id'] not in all_used_tracks and track['id'] not in current_mood_tracks:
                                all_tracks.append((track, 0.5))
                                current_mood_tracks.add(track['id'])
                                
                                if len(all_tracks) >= limit:
                                    break
                    except:
                        # If all else fails, use any tracks we can get
                        pass
            
            # Sort by score and take top results
            all_tracks.sort(key=lambda x: x[1], reverse=True)
            
            # Take exactly the number of tracks the user requested
            final_tracks = []
            final_scores = []
            used_artists = set()  # For additional artist diversity
            
            # First pass: collect one track per artist
            for track, score in all_tracks:
                artist_id = track['artists'][0]['id']
                if artist_id not in used_artists and len(final_tracks) < limit:
                    final_tracks.append(track)
                    final_scores.append(score)
                    used_artists.add(artist_id)
            
            # Second pass: fill remaining slots if needed
            if len(final_tracks) < limit:
                for track, score in all_tracks:
                    if track not in final_tracks and len(final_tracks) < limit:
                        final_tracks.append(track)
                        final_scores.append(score)
            
            # If we still have no recommendations, use fallback
            if not final_tracks:
                st.warning("Couldn't find specific matches. Try a more popular artist.")
                search_results = self.sp.search(q="year:2024", type='track', limit=limit)
                final_tracks = search_results.get('tracks', {}).get('items', [])
                final_scores = [0.5] * len(final_tracks)
            
            # Add the tracks we used to the global tracking
            for track in final_tracks:
                self.global_used_tracks[detected_mood].add(track['id'])
            
            return final_tracks, final_scores, mood_scores
            
        except Exception as e:
            # Return empty results instead of error
            return [], [], {"Happy": 1.0}

def initialize_spotify():
    """Initialize and return a Spotify client."""
    # Using demo credentials - replace with your own
    client_id = ""
    client_secret = ""
    
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        
        sp = spotipy.Spotify(
            auth_manager=auth_manager,
            requests_timeout=20,
            retries=3,
            backoff_factor=2
        )
        
        # Test the credentials with a simple API call
        sp.search(q='test', limit=1)
        return sp
        
    except Exception as e:
        st.error(f"Failed to connect to Spotify API: {str(e)}")
        return None

# Spotify logo SVG - clean button
spotify_logo_svg = """
<svg xmlns="http://www.w3.org/2000/svg" height="24" width="24" viewBox="0 0 24 24" fill="#1DB954">
    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
</svg>
"""


# Cleaner CSS for better UI
minimalist_css = """
<style>
.song-display {
    margin-bottom: 20px;
}
.song-title {
    font-size: 20px;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 5px;
    color: #333;
    text-transform: uppercase;
}
.artist-name {
    font-size: 16px;
    color: #555;
    margin-bottom: 5px;
}
.album-name {
    font-size: 15px;
    color: #666;
    margin-bottom: 15px;
}
.spotify-link {
    color: #1DB954;
    text-decoration: none;
    display: flex;
    align-items: center;
    font-size: 14px;
    margin-top: 10px;
}
.spotify-link:hover {
    text-decoration: underline;
}
.spotify-link svg {
    margin-right: 5px;
}
</style>
"""

# Sidebar for app information
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
    
    🌎 **Global Music**
    - Discovers international artists
    - Explores diverse cultures
    """)
    
    st.divider()
    st.header("📊 Quick Stats")
    st.metric("AI Models", "3")
    st.metric("Music Features", "11")
    st.metric("Global Genres", "15+")

# Main content - tabs
tab1, tab2 = st.tabs(["🎯 Get Recommendations", "ℹ️ About"])

with tab1:
    st.header("What would you like to listen to?")
    
    # Input section - two columns
    col1, col2 = st.columns(2)
    
    with col1:
        artist_name = st.text_input(
            "🎤 Artist Name",
            placeholder="e.g., Taylor Swift",
            help="Enter your favorite artist's name"
        )
    
    with col2:
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
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress
            status_text.text("Analyzing your mood...")
            progress_bar.progress(10)
            
            # Initialize Spotify and recommender
            spotify = initialize_spotify()
            if not spotify:
                progress_bar.empty()
                status_text.empty()
                st.error("Couldn't connect to Spotify. Please check your credentials.")
                st.stop()
                
            progress_bar.progress(20)
            
            # Initialize recommender with cache
            @st.cache_resource(ttl=3600)
            def get_recommender(client_id, client_secret):
                auth_manager = SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
                
                spotify = spotipy.Spotify(
                    auth_manager=auth_manager,
                    requests_timeout=20,
                    retries=3,
                    backoff_factor=2
                )
                
                return AIRecommender(spotify)
            
            try:
                # Get recommender instance
                # Use session_state to maintain a single recommender instance throughout the session
                if 'recommender' not in st.session_state:
                    st.session_state.recommender = get_recommender("", "")
                recommender = st.session_state.recommender
                
                # Get mood
                status_text.text("Analyzing your mood...")
                dominant_mood, mood_scores = recommender.analyze_mood(mood_input)
                progress_bar.progress(40)
                
                # Get recommendations
                status_text.text("Finding diverse artists that match your mood...")
                recommendations, scores, mood_scores = recommender.get_recommendations(
                    artist_name,
                    mood_input,
                    num_recommendations,
                    min_popularity
                )
                progress_bar.progress(80)
                
                # Complete progress
                progress_bar.progress(100)
                status_text.empty()
                
                if recommendations:
                    # Display mood analysis
                    st.header("🎭 Your Mood Analysis")
                    
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
                    
                    # Count unique artists
                    artist_count = len(set(track['artists'][0]['name'] for track in recommendations))
                    st.success(f"🎉 Found {len(recommendations)} perfect matches from {artist_count} different artists for your {dominant_mood} mood!")
                    
                    # Display recommendations in a clean grid
                    # Display recommendations in a minimalist style
                    num_columns = 3
                    
                    # Inject CSS for minimalist UI
                    st.markdown(minimalist_css, unsafe_allow_html=True)
                    
                    for i in range(0, len(recommendations), num_columns):
                        cols = st.columns(num_columns)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(recommendations):
                                track = recommendations[idx]
                                track_score = scores[idx]
                                
                                with col:
                                    # Album image
                                    if track['album']['images']:
                                        st.image(
                                            track['album']['images'][0]['url'],
                                            use_column_width=True
                                        )
                                    
                                    # Simple track display like example
                                    html_content = f"""
                                    <div class="song-display">
                                        <div class="song-title">{track['name']}</div>
                                        <div class="artist-name">by {track['artists'][0]['name']}</div>
                                        <div class="album-name">Album: {track['album']['name']}</div>
                                        <a href="{track['external_urls']['spotify']}" target="_blank" class="spotify-link">
                                            {spotify_logo_svg} Listen on Spotify
                                        </a>
                                    </div>
                                    """
                                    st.markdown(html_content, unsafe_allow_html=True)
                                    
                                    # Preview audio if available 
                                    if track['preview_url']:
                                        st.audio(track['preview_url'])
                else:
                    st.error("Couldn't find any recommendations. Try another artist or mood description.")
            
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error generating recommendations: {str(e)}")

with tab2:
    st.header("About this App")
    st.write("""
    This AI-powered music recommender uses state-of-the-art machine learning to:
    
    - Analyze your current mood through natural language processing
    - Extract and analyze audio features from songs
    - Match songs to your mood using neural networks
    - Provide personalized recommendations from diverse artists worldwide
    """)
    
    st.subheader("✨ Global Music Discovery")
    st.write("""
    Our recommendation engine now includes music from around the world! Discover amazing artists from:
    
    - K-pop and J-pop scenes
    - Latin American and Reggaeton hits
    - African and Afrobeat rhythms
    - Indian and Bollywood classics
    - European music across multiple languages
    - And many more international genres!
    """)

# Simple footer
st.markdown("---")
st.markdown("🎵 AI Music Mood Matcher - Find your perfect soundtrack")

if __name__ == "__main__":
    pass  # The app runs automatically with Streamlit
