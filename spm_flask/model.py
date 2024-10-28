import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk mendapatkan Vessel Group ID berdasarkan nama vessel
def get_vessel_group_id(df, vessel_name):
    vessel_row = df[df['VESSEL'] == vessel_name]
    if not vessel_row.empty:
        return vessel_row.iloc[0]['VESSEL GROUP ID']
    else:
        return None

# Helper function to calculate the difference between 'day_remains' and today
def calculate_day_remains_difference(df):
    # Convert the 'DAY REMAINS' column to datetime, handling errors
    df['DAY REMAINS'] = pd.to_datetime(df['DAY REMAINS'], format='%Y-%m-%d', errors='coerce')
    
    # Filter out rows where 'DAY REMAINS' couldn't be converted to a date
    df = df[df['DAY REMAINS'].notna()]
    
    # Calculate the difference in days
    today = datetime.now().date()
    df['DAY REMAINS'] = (df['DAY REMAINS'].dt.date - today).apply(lambda x: x.days if pd.notna(x) else None)

    return df

# Updated getRecommendation function
def getRecommendation(df, dataCandidates, bagian, vessel_name, rank, certificate, age_range):
    # Memfilter data berdasarkan BAGIAN dan VESSEL GROUP ID
    vessel_group_id = get_vessel_group_id(df, vessel_name)
    df_filtered = df[(df['BAGIAN'] == bagian) & (df['VESSEL GROUP ID'] == vessel_group_id)]

    # Memfilter berdasarkan umur
    df_filtered = df_filtered[(df_filtered['UMUR'] >= age_range[0]) & (df_filtered['UMUR'] <= age_range[1])]

    # Menggabungkan fitur RANK dan CERTIFICATE untuk perhitungan similarity
    df_filtered['combined_features'] = df_filtered['RANK'] + " " + df_filtered['CERTIFICATE']
    
    # Menghitung TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_filtered['combined_features'])
    
    # Membuat vektor dari input pengguna
    user_input = [rank + " " + certificate]
    user_vector = vectorizer.transform(user_input)
    
    # Menghitung cosine similarity antara input pengguna dan data
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Menambahkan similarity score ke data dan mengurutkan berdasarkan score
    df_filtered['similarity_score'] = similarity_scores
    df_filtered = df_filtered.sort_values(by='similarity_score', ascending=False)

    # Calculate day remains difference
    df_filtered = calculate_day_remains_difference(df_filtered)

    # Mengambil top 5 rekomendasi
    recommendations = df_filtered.head(20)[[
        'SEAMAN CODE', 'SEAFARER CODE', 'SEAMAN NAME', 'RANK', 'VESSEL', 
        'VESSEL GROUP ID', 'UMUR', 'CERTIFICATE', 'similarity_score', 'PHONE1', 'PHONE2', 'PHONE3', 'PHONE4', 'DAY REMAINS'
    ]]
    
    return recommendations

def search_candidate(df, bagian, vessel_name, age_range):
    # Periksa apakah 'VESSEL GROUP ID' ada di DataFrame
    if 'VESSEL GROUP ID' in df.columns:
        vessel_group_id = get_vessel_group_id(df, vessel_name)
        if vessel_group_id is None:
            return pd.DataFrame()  # Return empty DataFrame if no matching vessel is found
        
        # Memfilter data berdasarkan BAGIAN, VESSEL GROUP ID, dan umur
        filtered_data = df[
            (df['BAGIAN'] == bagian) & 
            (df['VESSEL GROUP ID'] == vessel_group_id) & 
            (df['UMUR'] >= age_range[0]) & 
            (df['UMUR'] <= age_range[1])
        ]
    else:
        # Jika 'VESSEL GROUP ID' tidak ada, hanya filter berdasarkan BAGIAN dan umur
        filtered_data = df[
            (df['BAGIAN'] == bagian) & 
            (df['UMUR'] >= age_range[0]) & 
            (df['UMUR'] <= age_range[1])
        ]
    
    return filtered_data


