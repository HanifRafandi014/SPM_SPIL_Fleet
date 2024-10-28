from flask import Flask, session, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
from io import BytesIO
from model import getRecommendation, search_candidate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey' 

SHIP_GROUPS = {
    'manalagi_rotation': ["KM. MANALAGI PRITA", "KM. MANALAGI ASTA", "KM. MANALAGI ASTI", "KM. MANALAGI DASA", "KM. MANALAGI ENZI", "KM. MANALAGI TARA", "KM. MANALAGI WANDA"],
    'manalagi_rotation2': ["KM. MANALAGI TISYA", "KM. MANALAGI SAMBA", "KM. MANALAGI HITA", "KM. MANALAGI VIRA", "KM. MANALAGI YASA", "KM. XYS SATU"],
    'manalagi_kkm' : ["KM. MANALAGI ASTA", "KM. MANALAGI ASTI", "KM. MANALAGI SAMBA", "KM. MANALAGI YASA", "KM. XYS SATU", "KM. MANALAGI WANDA"],
    'manalagi_kkm2' : ["KM. MANALAGI TISYA", "KM. MANALAGI PRITA", "KM. MANALAGI DASA", "KM. MANALAGI HITA", "KM. MANALAGI ENZI", "KM. MANALAGI TARA", "KM. MANALAGI VIRA"],
    'container_rotation1' : ["KM. ORIENTAL EMERALD", "KM. ORIENTAL RUBY", "KM. ORIENTAL SILVER", "KM. ORIENTAL GOLD", "KM. ORIENTAL JADE", "KM. ORIENTAL DIAMOND"],
    'container_rotation2' : ["KM. LUZON", "KM. VERIZON", "KM. ORIENTAL GALAXY", "KM. HIJAU SAMUDRA", "KM. ARMADA PERMATA"],
    'container_rotation3' : ["KM. ORIENTAL SAMUDERA", "KM. ORIENTAL PACIFIC", "KM. PULAU NUNUKAN", "KM. TELUK FLAMINGGO", "KM. TELUK BERAU", "KM. TELUK BINTUNI"],
    'container_rotation4' : ["KM. PULAU LAYANG", "KM. PULAU WETAR", "KM. PULAU HOKI", "KM. SPIL HANA", "KM. SPIL HASYA", "KM. SPIL HAPSRI", "KM. SPIL HAYU"],
    'container_rotation5' : ["KM. HIJAU JELITA", "KM. HIJAU SEJUK", "KM. ARMADA SEJATI", "KM. ARMADA SERASI", "KM. ARMADA SEGARA", "KM. ARMADA SENADA", "KM. HIJAU SEGAR", "KM. TITANIUM", "KM. VERTIKAL"],
    'container_rotation6' : ["KM. SPIL RENATA", "KM. SPIL RATNA", "KM. SPIL RUMI", "KM. PEKAN BERAU", "KM SPIL RAHAYU", "KM. SPIL RETNO", "KM. MINAS BARU", "KM PEKAN SAMPIT", "KM. SELILI BARU"],
    'container_rotation7' : ["KM. DERAJAT", "KM. MULIANIM", "KM. PRATIWI RAYA", "KM. MAGELLAN", "KM. PAHALA", "KM. PEKAN RIAU", "KM. PEKAN FAJAR", "KM. FORTUNE"],
    'container_rotation8' : ["KM. PRATIWI SATU", "KM. BALI SANUR", "KM. BALI KUTA", "KM. BALI GIANYAR", "KM. BALI AYU", "KM. AKASHIA", "KM KAPPA"],
    'container_kkm1' : ["KM. ORIENTAL GOLD", "KM. ORIENTAL EMERALD", "KM. ORIENTAL GALAXY", "KM. ORIENTAL RUBY", "KM. ORIENTAL SILVER", "KM. ORIENTAL JADE", "KM. VERIZON", "KM. LUZON", "KM. ORIENTAL DIAMOND"],
    'container_kkm2' : ["KM. SPIL HAPSRI", "KM. ARMADA PERMATA", "KM. HIJAU SAMUDRA", "KM. SPIL HASYA", "KM. ARMADA SEJATI", "KM. SPIL HAYU", "KM. SPIL HANA", "KM. HIJAU SEJUK", "KM. HIJAU JELITA"],
    'container_kkm3' : ["KM. ORIENTAL PACIFIC", "KM. ORIENTAL SAMUDERA", "KM. ARMADA SEGARA", "KM. ARMADA SENADA", "KM. ARMADA SERASI", "KM. SPIL RATNA", "KM. SPIL RUMI", "KM. PULAU NUNUKAN"],
    'container_kkm4' : ["KM. PULAU HOKI", "KM. TELUK BINTUNI", "KM. TELUK FLAMINGGO", "KM. PULAU LAYANG", "KM. TELUK BERAU", "KM. SPIL RENATA", "KM. PULAU WETAR", "KM SPIL RAHAYU", "KM. SPIL RETNO"],
    'container_kkm5' : ["KM. MINAS BARU", "KM. SELILI BARU", "KM. VERTIKAL", "KM. HIJAU SEGAR", "KM. PEKAN RIAU", "KM. PEKAN BERAU", "KM. PEKAN FAJAR", "KM. PEKAN SAMPIT", "KM. TITANIUM"],
    'container_kkm6' : ["KM. PRATIWI RAYA", "KM. PRATIWI SATU", "KM. BALI AYU", "KM. BALI GIANYAR", "KM. BALI SANUR", "KM. BALI KUTA"],
    'container_kkm7' : ["KM. MAGELLAN", "KM. MULIANIM", "KM. PAHALA", "KM. FORTUNE", "KM. AKASHIA", "KM. DERAJAT"]
}

# Load the Excel sheet into a dataframe
df = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='Container-Deck')
df2 = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='Container-Engine')
df3 = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='manalagi-Deck')
df4 = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='manalagi-Engine')
df5 = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='BC')
df6 = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='MT')
df7 = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='TK')
df8 = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='TB')
df9 = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='others')

# Load the seamen data
combined_df = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='List of Active Seamen')

# Get today's date
today = pd.Timestamp(datetime.datetime.now().date())

# Convert 'DAY REMAINS' to datetime
combined_df['DAY REMAINS'] = pd.to_datetime(combined_df['DAY REMAINS'], errors='coerce')

# Calculate the difference between 'DAY REMAINS' and today's date
combined_df['DAY REMAINS DIFF'] = (combined_df['DAY REMAINS'] - today).dt.days

# Filter the rows where 'DAY REMAINS DIFF' > 0
df_filtered = combined_df[combined_df['DAY REMAINS DIFF'] > 0][['SEAMAN CODE', 'SEAFARER CODE', 'SEAMAN NAME', 'RANK', 'VESSEL', 'UMUR', 'CERTIFICATE', 'DAY REMAINS DIFF']]

# Sort the data by 'DAY REMAINS DIFF' in ascending order
sorted_df = df_filtered.sort_values(by='DAY REMAINS DIFF')

# Save the sorted dataframe for display in the HTML table
sorted_df.to_csv('./data/sorted_seamen_data_diff.csv', index=False)

# Function to get the top 5 similar seamen based on rank, certificate, and vessel
def get_top_5_similar(target_seaman_code):
    try:
        # Filter target seaman by SEAMAN CODE
        target_seaman = df_filtered[df_filtered['SEAMAN CODE'] == target_seaman_code]
        
        # Check if the target seaman exists
        if target_seaman.empty:
            return {"error": "Seaman not found"}, 404  # Return an error if the seaman is not found
        
        # Preprocess 'RANK', 'CERTIFICATE', and 'VESSEL' columns, fill NaN with empty strings
        df_filtered['rank_certificate_vessel_combined'] = df_filtered[['RANK', 'CERTIFICATE', 'VESSEL']].apply(
            lambda x: ' '.join(x.fillna('').replace({'[^A-Za-z0-9 ]+': ''})), axis=1  # Removing special characters
        )
        
        # Get target's combined fields (RANK, CERTIFICATE, VESSEL) for the target seaman
        target_combined = target_seaman[['RANK', 'CERTIFICATE', 'VESSEL']].apply(
            lambda x: ' '.join(x.fillna('').replace({'[^A-Za-z0-9 ]+': ''})), axis=1
        ).values[0]

        # Use TF-IDF vectorizer for cosine similarity
        # Now fit the vectorizer including the target_combined string to calculate the similarity
        vectorizer = TfidfVectorizer()
        
        # Fit the vectorizer with all combined data including the target's combined fields
        combined_data = df_filtered['rank_certificate_vessel_combined'].tolist()
        combined_data.insert(0, target_combined)  # Insert the target_combined at the start

        # Transform the combined data
        tfidf_matrix = vectorizer.fit_transform(combined_data)
        
        # Compute cosine similarities between the target and all others
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()

        # Remove the first similarity score (which is the target comparing with itself)
        similarity_scores = list(enumerate(cosine_similarities))[1:]
        
        # Sort the similarity scores to get the top 5 most similar seamen
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]
        
        # Get the indices of the top 5 similar seamen
        top_5_indices = [i[0] - 1 for i in similarity_scores]  # Subtract 1 because we inserted target_combined at index 0

        return df_filtered.iloc[top_5_indices].to_dict(orient='records')

    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": str(e)}, 500  # Catch any errors and return a 500 response with the error message

# Route to serve the main dashboard
@app.route('/')
def dashboard():
    # Load the processed data directly in the route
    data = pd.read_csv('./data/sorted_seamen_data_diff.csv')  # Replace with the actual path to the CSV file
    return render_template('dashboard.html', data=data.to_dict(orient='records'))

# Route to get the top 5 similar seamen
@app.route('/similarity/<int:seaman_code>', methods=['GET'])
def get_similarity(seaman_code):
    top_5 = get_top_5_similar(seaman_code)
    return jsonify(top_5)

# Global variable to hold the current DataFrame
current_df = df

GROUP_ID_MAP = {
    'manalagi_rotation': 1,
    'manalagi_rotation2': 2
}

GROUP_ID_MAP2 = {
    'manalagi_kkm': 1,
    'manalagi_kkm2': 2
}

GROUP_ID_MAP3 = {
    'container_rotation1': 1,
    'container_rotation2': 2,
    'container_rotation3': 3,
    'container_rotation4': 4,
    'container_rotation5': 5,
    'container_rotation6': 6,
    'container_rotation7': 7,
    'container_rotation8': 8
}

GROUP_ID_MAP4 = {
    'container_kkm1': 1,
    'container_kkm2': 2,
    'container_kkm3': 3,
    'container_kkm4': 4,
    'container_kkm5': 5,
    'container_kkm6': 6,
    'container_kkm7': 7
}

def generate_schedule(ship_names, first_assignments, start_year, end_year):
    months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    # Calculate total months based on start and end year
    total_months = (end_year - start_year + 1) * 12

    # Generate headers for the schedule based on the range of years
    headers = []
    for i in range(total_months):
        month_name = months[i % 12]
        current_year = start_year + (i // 12)
        headers.append(f"{month_name} {current_year}")
    
    # Create the schedule DataFrame with ship names as index
    schedule = pd.DataFrame(columns=headers, index=ship_names)
    
    crew = [f'C{i+1}' for i in range(len(ship_names) + 1)]
    sorted_assignments = sorted(enumerate(first_assignments, start=1), key=lambda x: x[1]['month'])

    for i, (ship_idx, assignment) in enumerate(sorted_assignments):
        start_month = (assignment['year'] - start_year) * 12 + assignment['month'] - 1
        ship_name = ship_names[ship_idx - 1]
        crew_idx = i % len(crew)

        # Assign initial crew based on start month
        for j in range(len(ship_names)):
            current_month = (start_month + j) % total_months
            schedule.iloc[ship_idx-1, current_month] = crew[crew_idx]

        # Assign transaction crew after initial period
        transaction_crew_idx = (crew_idx - 1) % len(crew)
        last_transaction_month = start_month

        for j in range(start_month + len(ship_names), total_months, len(ship_names)):
            current_month = j % total_months
            schedule.iloc[ship_idx-1, current_month] = f"{crew[transaction_crew_idx]} (transaction)"
            
            # After each transaction, backfill NaN months between this and the previous transaction
            for k in range(last_transaction_month + 1, current_month):
                if pd.isna(schedule.iloc[ship_idx-1, k]):
                    schedule.iloc[ship_idx-1, k] = crew[crew_idx]

            last_transaction_month = current_month
            crew_idx = transaction_crew_idx
            transaction_crew_idx = (transaction_crew_idx - 1) % len(crew)

        # After the final transaction, fill the remaining NaN months
        for k in range(last_transaction_month + 1, total_months):
            if pd.isna(schedule.iloc[ship_idx-1, k]):
                schedule.iloc[ship_idx-1, k] = crew[crew_idx]

    return schedule

# kode 1
def prepare_display_df(df): 
    display_df = df[['SEAMAN CODE', 'SEAMAN NAME', 'VESSEL GROUP ID', 'RANK', 'CERTIFICATE']].copy() 
    display_df.rename(columns={'UMUR': 'AGE'}, inplace=True)
    return display_df

def prioritize_nakhoda_ant1(df):
    certificate_priority = {
        'ANT-I': 5,
        'ANT-II': 4,
        'ANT-III': 3,
        'ANT-IV': 2,
        'ANT-D': 1
    }
    
    df['priority'] = df['CERTIFICATE'].map(certificate_priority).fillna(0)  
    
    sorted_df = df.sort_values(by='priority', ascending=False).drop(columns=['priority'])
    return sorted_df

def filter_group_1(df, group_id):
    filtered_df = df[df['VESSEL GROUP ID'] == group_id].copy()
    return filtered_df

# kode 2
def prepare_display_df2(df): 
    display_df2 = df[['SEAMAN CODE', 'SEAMAN NAME', 'VESSEL GROUP ID', 'RANK', 'CERTIFICATE']].copy() 
    display_df2.rename(columns={'UMUR': 'AGE'}, inplace=True)
    return display_df2

def prioritize_nakhoda_ant2(df):
    certificate_priority = {
        'ATT-I': 6,
        'ATT-II': 5,
        'ATT-III': 4,
        'ATT-IV': 3,
        'ATT-V': 2,
        'ATT-D': 1
    }
    
    df['priority'] = df['CERTIFICATE'].map(certificate_priority).fillna(0)  
    
    sorted_df2 = df.sort_values(by='priority', ascending=False).drop(columns=['priority'])
    return sorted_df2

def filter_group_2(df, group_id):
    filtered_df2 = df[df['VESSEL GROUP ID'] == group_id].copy()
    return filtered_df2

def generate_crew_backup_pairs(ship_names, first_assignments):
    crew = [f'C{i+1}' for i in range(len(ship_names) + 1)]
    backup_pairs = []
    
    # Determine backup pairs based on first assignment and transaction logic
    for i in range(len(ship_names)+1):
        main_crew = crew[i]  # Crew utama sesuai urutan
        backup_crew = crew[(i - 1) % len(crew)]  # Backup mengikuti aturan rotasi mundur

        # Mengatasi kasus rotasi C1 digantikan oleh C7
        if i == 0:
            backup_crew = crew[-1]  # C1 digantikan oleh C7

        backup_pairs.append({'main': main_crew, 'backup': backup_crew})
    
    return backup_pairs

@app.route('/container_rotation', methods=['GET', 'POST'])
def container_rotation():
    if request.method == 'POST':
        try:
            # Part 1: Handle ship group and scheduling
            ship_group = request.form['ship_group']
            ship_names = SHIP_GROUPS[ship_group]  # Get the ship names from the selected group
            num_ships = len(ship_names)  # Get the number of ships based on the group

            # Extract start and end year from form
            start_year = int(request.form['start_year'])
            end_year = int(request.form['end_year'])

            # Get first assignments as a list of dictionaries with month and year
            first_assignments = [
                {
                    'month': int(request.form[f'first_assignments[{i}][month]']),
                    'year': int(request.form[f'first_assignments[{i}][year]'])
                }
                for i in range(1, num_ships + 1)
            ]

            # Generate the schedule using the ship names, first assignments, start year, and end year
            schedule = generate_schedule(ship_names, first_assignments, start_year, end_year)
            schedule_html = schedule.to_html(classes='table table-striped table-bordered', na_rep='')

            # Part 2: Handle seamen data processing
            container_rotation_df = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='Container-Captain')  # Load data

            container_rotation_filtered = prepare_display_df(container_rotation_df)

            # Prioritize and filter the data
            container_rotation_prioritized = prioritize_nakhoda_ant1(container_rotation_filtered)
            group_id = GROUP_ID_MAP3.get(ship_group, 1)
            container_rotation_group1 = filter_group_1(container_rotation_prioritized, group_id)

            # Add 'Transaction' column with labels 'C1', 'C2', ..., based on the number of recommendations
            container_rotation_group1['CODE'] = ['C' + str(i + 1) for i in range(len(container_rotation_group1))]

            # Now you can select the required columns, including 'CODE'
            container_rotation_recommendations = container_rotation_group1[
                ['SEAMAN CODE', 'SEAMAN NAME', 'VESSEL GROUP ID', 'RANK', 'CERTIFICATE', 'CODE']
            ].head((num_ships + 2) * 2)

            # Step 3: Split recommendations into inti, cad, and sisa lists
            inti_size = num_ships
            cad_size = 1  # One reserved candidate

            # Split recommendations into primary candidates (inti), reserved candidate (cad), and remaining candidates (sisa)
            inti = container_rotation_recommendations.iloc[:inti_size].to_dict('records')
            cad = container_rotation_recommendations.iloc[inti_size:inti_size + cad_size].to_dict('records')
            sisa = container_rotation_recommendations.iloc[inti_size + cad_size:].to_dict('records')

            crew_backup_pairs = generate_crew_backup_pairs(ship_names, first_assignments)

            # Save data into session for later use (e.g., CSV download)
            session['schedule'] = schedule.to_dict()
            session['inti'] = inti
            session['cad'] = cad
            session['sisa'] = sisa
            session['crew_backup_pairs'] = crew_backup_pairs
            session['ship_names'] = ship_names  # Save ship names
            session['start_year'] = start_year  # Save start and end year for generating headers
            session['end_year'] = end_year

            return render_template('container_rotation.html',
                                   schedule_html=schedule_html, 
                                   inti=inti,  # Primary candidates
                                   cad=cad,    # Reserved candidate
                                   sisa=sisa, crew_backup_pairs=crew_backup_pairs)  # Remaining candidates
        except Exception as e:
            return f"An error occurred: {e}"

    return render_template('container_rotation.html')

@app.route('/manalagi_rotation', methods=['GET', 'POST'])
def manalagi_rotation():
    if request.method == 'POST':
        try:
            # Part 1: Handle ship group and scheduling logic
            ship_group = request.form['ship_group']
            ship_names = SHIP_GROUPS[ship_group]  # Get the ship names based on selected group
            num_ships = len(ship_names)

            # Extract start and end years from the form
            start_year = int(request.form['start_year'])
            end_year = int(request.form['end_year'])

            # Get first assignments as a list of dictionaries from the form
            first_assignments = [
                {
                    'month': int(request.form[f'first_assignments[{i}][month]']),
                    'year': int(request.form[f'first_assignments[{i}][year]'])
                }
                for i in range(1, num_ships + 1)
            ]

            # Generate the schedule using ship names, first assignments, and start/end years
            schedule = generate_schedule(ship_names, first_assignments, start_year, end_year)
            schedule_html = schedule.to_html(classes='table table-striped table-bordered', na_rep='')

            # Part 2: Handle seamen data processing (read from Excel or other source)
            manalagi_rotation_df = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='Manalagi-Captain')  # Load data from Excel

            # Filter and prepare the display dataframe
            manalagi_rotation_filtered = prepare_display_df(manalagi_rotation_df)

            # Prioritize and filter the data
            manalagi_rotation_prioritized = prioritize_nakhoda_ant1(manalagi_rotation_filtered)
            group_id = GROUP_ID_MAP.get(ship_group, 1)
            manalagi_rotation_group1 = filter_group_1(manalagi_rotation_prioritized, group_id)

            # **Add 'CODE' column for labeling ('C1', 'C2', ..., 'RC')**
            manalagi_rotation_group1['CODE'] = ['C' + str(i + 1) for i in range(len(manalagi_rotation_group1))]

            # Now you can select the required columns, including 'CODE'
            manalagi_rotation_recommendations = manalagi_rotation_group1[
                ['SEAMAN CODE', 'SEAMAN NAME', 'VESSEL GROUP ID', 'RANK', 'CERTIFICATE', 'CODE']
            ].head((num_ships + 2) * 2)

            # Step 3: Split recommendations into inti, cad, and sisa lists
            inti_size = num_ships
            cad_size = 1  # One reserved candidate

            # Split recommendations into primary candidates (inti), reserved candidate (cad), and remaining candidates (sisa)
            inti = manalagi_rotation_recommendations.iloc[:inti_size].to_dict('records')
            cad = manalagi_rotation_recommendations.iloc[inti_size:inti_size + cad_size].to_dict('records')
            sisa = manalagi_rotation_recommendations.iloc[inti_size + cad_size:].to_dict('records')

            crew_backup_pairs = generate_crew_backup_pairs(ship_names, first_assignments)

            # Save data into session for later use (e.g., CSV download)
            session['schedule'] = schedule.to_dict()
            session['inti'] = inti
            session['cad'] = cad
            session['sisa'] = sisa
            session['crew_backup_pairs'] = crew_backup_pairs
            session['ship_names'] = ship_names  # Save ship names
            session['start_year'] = start_year  # Save start and end year for generating headers
            session['end_year'] = end_year

            # Render the template and pass the processed data (inti, cad, sisa) directly
            return render_template('manalagi_rotation.html',
                                   schedule_html=schedule_html,
                                   inti=inti,  # Primary candidates
                                   cad=cad,    # Reserved candidate
                                   sisa=sisa, crew_backup_pairs=crew_backup_pairs)  # Remaining candidates
        except Exception as e:
            # In case of an error, return the error message
            return f"An error occurred: {e}"

    # If GET request, just render the form for selecting ship group, start and end year
    return render_template('manalagi_rotation.html')

@app.route('/drop', methods=['POST'])
def drop():
    try:
        # Retrieve lists from session
        inti = session.get('inti', [])
        cad = session.get('cad', [])
        sisa = session.get('sisa', [])

        # Get the SEAMAN CODE of the person to be dropped
        karyawan_dropped_code = request.form['karyawan']

        # print(f"Dropping: {karyawan_dropped_code}")

        # Find the index of the dropped seaman in 'inti'
        posisi = next((i for i, k in enumerate(inti) if k['SEAMAN CODE'] == int(karyawan_dropped_code)), None)

        if posisi is not None:
            # Move the dropped employee to 'sisa'
            dropped_employee = inti.pop(posisi)
            sisa.append(dropped_employee)

            # print(f"Moved to sisa: {dropped_employee}")

            # Replace the dropped employee with the first reserved candidate from 'cad'
            if cad:
                inti.insert(posisi, cad[0])  # Insert the first cadangan in the same position
                # print(f"Replaced with cadangan: {cad[0]}")

                # Update 'cad' to have the next candidate from 'sisa' (if available)
                if sisa:
                    cad[0] = sisa.pop(0)  # Set the next person from 'sisa' as the new 'cad'
                    # print(f"New cadangan: {cad[0]}")
                else:
                    cad.clear()  # No more reserve candidates, clear 'cad'
                    # print("No more cadangan left.")
            else:
                print("No cadangan to replace.")

        # Update session lists with the modified 'inti', 'cad', and 'sisa'
        session['inti'] = inti
        session['cad'] = cad
        session['sisa'] = sisa

        # Return the updated data in JSON format
        return {'inti': inti, 'sisa': sisa, 'cad': cad}

    except Exception as e:
        return f"An error occurred in drop: {e}"

    
@app.route('/change', methods=['POST'])
def change():
    try:
        # Retrieve lists from session
        inti = session.get('inti', [])
        cad = session.get('cad', [])
        sisa = session.get('sisa', [])

        # Retrieve the form data
        karyawan_to_change_code = str(request.form['karyawan']).strip()  # SEAMAN CODE of the employee to change
        karyawan_replacement_code = str(request.form['replacement']).strip()  # SEAMAN CODE of the replacement employee

        # Ensure that all SEAMAN CODE values in 'inti' are strings and strip any whitespace
        inti = [{**k, 'SEAMAN CODE': str(k['SEAMAN CODE']).strip()} for k in inti]

        # Find the employee to change in 'inti'
        inti_index = next((i for i, k in enumerate(inti) if k['SEAMAN CODE'] == karyawan_to_change_code), None)
        if inti_index is None:
            return {'error': 'The selected employee to change was not found in the Inti list.'}, 400

        # Ensure that all SEAMAN CODE values in 'sisa' are strings and strip any whitespace
        sisa = [{**k, 'SEAMAN CODE': str(k['SEAMAN CODE']).strip()} for k in sisa]

        # Find the replacement in 'sisa'
        replacement = next((k for k in sisa if k['SEAMAN CODE'] == karyawan_replacement_code), None)
        if replacement is None:
            return {'error': 'The selected replacement was not found in the Sisa list.'}, 400

        # Replace the employee in 'inti'
        old_employee = inti[inti_index]
        inti[inti_index] = replacement  # Replace with the selected replacement from 'sisa'

        # Update 'sisa' and add the old employee to 'sisa'
        sisa = [k for k in sisa if k['SEAMAN CODE'] != karyawan_replacement_code]  # Remove replacement from 'sisa'
        sisa.append(old_employee)  # Add the old employee to 'sisa'

        # Update session lists
        session['inti'] = inti
        session['sisa'] = sisa

        # Ensure that the response contains the replacement object and row index
        return {
            'replacement': {
                'SEAMAN CODE': replacement['SEAMAN CODE'],
                'SEAMAN NAME': replacement['SEAMAN NAME'],
                'VESSEL GROUP ID': replacement['VESSEL GROUP ID'],
                'RANK': replacement['RANK'],
                'CERTIFICATE': replacement['CERTIFICATE'],
                'CODE': replacement['CODE']
            },
            'row_index': inti_index  # Send the index of the changed row
        }
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/container_kkm', methods=['GET', 'POST'])
def container_kkm():
    if request.method == 'POST':
        try:
            # Part 1: Handle ship group and scheduling
            ship_group = request.form['ship_group']
            ship_names = SHIP_GROUPS[ship_group]  # Get the ship names from the selected group
            num_ships = len(ship_names)  # Get the number of ships based on the group

            # Extract start and end year from form
            start_year = int(request.form['start_year'])
            end_year = int(request.form['end_year'])

            # Get first assignments as a list of dictionaries with month and year
            first_assignments = [
                {
                    'month': int(request.form[f'first_assignments[{i}][month]']),
                    'year': int(request.form[f'first_assignments[{i}][year]'])
                }
                for i in range(1, num_ships + 1)
            ]

            # Generate the schedule using the ship names, first assignments, start year, and end year
            schedule = generate_schedule(ship_names, first_assignments, start_year, end_year)
            schedule_html = schedule.to_html(classes='table table-striped table-bordered', na_rep='')

            # Part 2: Handle seamen data processing
            container_KKM_df = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='Container-KKM')  # Load data

            container_KKM_filtered = prepare_display_df(container_KKM_df)

            # Prioritize and filter the data
            container_KKM_prioritized = prioritize_nakhoda_ant1(container_KKM_filtered)
            group_id = GROUP_ID_MAP4.get(ship_group, 1)
            container_KKM_group1 = filter_group_1(container_KKM_prioritized, group_id)

            # Add 'Transaction' column with labels 'C1', 'C2', ..., based on the number of recommendations
            container_KKM_group1['CODE'] = ['C' + str(i + 1) for i in range(len(container_KKM_group1))]

            # Now you can select the required columns, including 'CODE'
            container_KKM_recommendations = container_KKM_group1[
                ['SEAMAN CODE', 'SEAMAN NAME', 'VESSEL GROUP ID', 'RANK', 'CERTIFICATE', 'CODE']
            ].head((num_ships + 2) * 2)

            # Step 3: Split recommendations into inti, cad, and sisa lists
            inti_size = num_ships
            cad_size = 1  # One reserved candidate

            # Split recommendations into primary candidates (inti), reserved candidate (cad), and remaining candidates (sisa)
            inti = container_KKM_recommendations.iloc[:inti_size].to_dict('records')
            cad = container_KKM_recommendations.iloc[inti_size:inti_size + cad_size].to_dict('records')
            sisa = container_KKM_recommendations.iloc[inti_size + cad_size:].to_dict('records')

            crew_backup_pairs = generate_crew_backup_pairs(ship_names, first_assignments)

            # Save data into session for later use (e.g., CSV download)
            session['schedule'] = schedule.to_dict()
            session['inti'] = inti
            session['cad'] = cad
            session['sisa'] = sisa
            session['crew_backup_pairs'] = crew_backup_pairs
            session['ship_names'] = ship_names  # Save ship names
            session['start_year'] = start_year  # Save start and end year for generating headers
            session['end_year'] = end_year

            return render_template('container_kkm.html',
                                   schedule_html=schedule_html, 
                                   inti=inti,  # Primary candidates
                                   cad=cad,    # Reserved candidate
                                   sisa=sisa, crew_backup_pairs=crew_backup_pairs)  # Remaining candidates
        except Exception as e:
            return f"An error occurred: {e}"

    return render_template('container_kkm.html')

@app.route('/manalagi_kkm', methods=['GET', 'POST'])
def manalagi_kkm():
    if request.method == 'POST':
        try:
            # Part 1: Handle ship group and scheduling
            ship_group = request.form['ship_group']
            ship_names = SHIP_GROUPS[ship_group]  # Get the ship names from the selected group
            num_ships = len(ship_names)  # Get the number of ships based on the group

            # Extract start and end year from form
            start_year = int(request.form['start_year'])
            end_year = int(request.form['end_year'])

            # Get first assignments as a list of dictionaries with month and year
            first_assignments = [
                {
                    'month': int(request.form[f'first_assignments[{i}][month]']),
                    'year': int(request.form[f'first_assignments[{i}][year]'])
                }
                for i in range(1, num_ships + 1)
            ]

            # Generate the schedule using the ship names, first assignments, start year, and end year
            schedule = generate_schedule(ship_names, first_assignments, start_year, end_year)
            schedule_html = schedule.to_html(classes='table table-striped table-bordered', na_rep='')

            # Part 2: Handle seamen data processing
            manalagi_KKM_df = pd.read_excel("./data/Seamen Report_rev1.xlsx", sheet_name='Manalagi-KKM')  # Load data

            manalagi_KKM_filtered = prepare_display_df(manalagi_KKM_df)

            # Prioritize and filter the data
            manalagi_KKM_prioritized = prioritize_nakhoda_ant1(manalagi_KKM_filtered)
            group_id = GROUP_ID_MAP2.get(ship_group, 1)
            manalagi_KKM_group1 = filter_group_1(manalagi_KKM_prioritized, group_id)

            # Add 'Transaction' column with labels 'C1', 'C2', ..., and 'RC'
            manalagi_KKM_group1['CODE'] = ['C' + str(i + 1) for i in range(len(manalagi_KKM_group1))]

            # Now you can select the required columns, including 'CODE'
            manalagi_KKM_recommendations = manalagi_KKM_group1[
                ['SEAMAN CODE', 'SEAMAN NAME', 'VESSEL GROUP ID', 'RANK', 'CERTIFICATE', 'CODE']
            ].head((num_ships + 2) * 2)

            # Step 3: Split recommendations into inti, cad, and sisa lists
            inti_size = num_ships
            cad_size = 1  # One reserved candidate

            # Split recommendations into primary candidates (inti), reserved candidate (cad), and remaining candidates (sisa)
            inti = manalagi_KKM_recommendations.iloc[:inti_size].to_dict('records')
            cad = manalagi_KKM_recommendations.iloc[inti_size:inti_size + cad_size].to_dict('records')
            sisa = manalagi_KKM_recommendations.iloc[inti_size + cad_size:].to_dict('records')

            crew_backup_pairs = generate_crew_backup_pairs(ship_names, first_assignments)

            # Save data into session for later use (e.g., CSV download)
            session['schedule'] = schedule.to_dict()
            session['inti'] = inti
            session['cad'] = cad
            session['sisa'] = sisa
            session['crew_backup_pairs'] = crew_backup_pairs
            session['ship_names'] = ship_names  # Save ship names
            session['start_year'] = start_year  # Save start and end year for generating headers
            session['end_year'] = end_year

            return render_template('manalagi_kkm.html',
                                   schedule_html=schedule_html, 
                                   inti=inti,  # Primary candidates
                                   cad=cad,    # Reserved candidate
                                   sisa=sisa, crew_backup_pairs=crew_backup_pairs)  # Remaining candidates
        except Exception as e:
            return f"An error occurred: {e}"

    return render_template('manalagi_kkm.html')

def generate_ordered_headers(start_year, end_year):
    """
    Function to generate ordered headers starting from January of start_year to December of end_year.
    """
    months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    headers = []

    for year in range(start_year, end_year + 1):
        for month in months:
            headers.append(f"{month} {year}")

    return headers

@app.route('/download_csv', methods=['GET'])
def download_csv():
    try:
        # Retrieve tables from session
        schedule = session.get('schedule', {})
        inti = session.get('inti', [])
        cad = session.get('cad', [])
        crew_backup_pairs = session.get('crew_backup_pairs', [])
        start_year = session.get('start_year')
        end_year = session.get('end_year')

        # Generate ordered headers (from Januari 2024 to Desember 2025, for example)
        ordered_headers = generate_ordered_headers(start_year, end_year)

        # Convert schedule to DataFrame
        schedule_df = pd.DataFrame(schedule)
        # print(schedule_df)

        # Ensure all columns in schedule_df have the same length
        for header in ordered_headers:
            if header not in schedule_df.columns:
                # Add missing columns and fill with empty strings
                schedule_df[header] = [''] * len(schedule_df)

        # Reorder columns based on the ordered headers for "Crew Rotation Table"
        schedule_df = schedule_df.reindex(columns=ordered_headers, fill_value='')

        # Sort rows by the "C" code (assuming the first column contains the codes such as C1, C2, C3, etc.)
        schedule_df.sort_values(by=schedule_df.columns[0], inplace=True)

        # Define the correct column order for "Manalagi Rotation Inti" and "Manalagi Rotation Cadangan"
        inti_columns_order = ['SEAMAN CODE', 'SEAMAN NAME', 'VESSEL GROUP ID', 'RANK', 'CERTIFICATE', 'CODE', 'ACTIONS']
        cad_columns_order = inti_columns_order  # Same order for cadangan

        # Reorder columns for inti and cad DataFrames
        inti_df = pd.DataFrame(inti).reindex(columns=inti_columns_order, fill_value='')
        cad_df = pd.DataFrame(cad).reindex(columns=cad_columns_order, fill_value='')

        # Ensure the same length in all columns
        for df in [inti_df, cad_df]:
            max_length = max(len(df[col]) for col in df.columns)
            for col in df.columns:
                if len(df[col]) < max_length:
                    df[col] = df[col].tolist() + [''] * (max_length - len(df[col]))

        # Define the correct column order for "Crew and Backup"
        crew_backup_columns_order = ['main', 'backup']

        # Reorder columns for crew backup DataFrame
        crew_backup_df = pd.DataFrame(crew_backup_pairs).reindex(columns=crew_backup_columns_order, fill_value='')

        # Create an Excel writer object
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write each table to a separate sheet in the correct order
            schedule_df.to_excel(writer, sheet_name='Crew Rotation Table', index=False)
            inti_df.to_excel(writer, sheet_name='Manalagi Rotation Inti', index=False)
            cad_df.to_excel(writer, sheet_name='Manalagi Rotation Cadangan', index=False)
            crew_backup_df.to_excel(writer, sheet_name='Crew and Backup', index=False)

        # Save the file to a BytesIO object
        output.seek(0)

        # Send the file back to the client
        return send_file(output, download_name="manalagi_rotation_ordered.xlsx", as_attachment=True)
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/container_deck')
def container_deck():
    global current_df
    current_df = df
    # Dapatkan nilai "Bagian" dari kolom BAGIAN di DataFrame
    bagian_value = current_df['BAGIAN'].iloc[0] if 'BAGIAN' in current_df.columns else 'Deck'
    return render_template('container_deck.html', bagian=bagian_value)

@app.route('/container_engine')
def container_engine():
    global current_df
    current_df = df2
    # Dapatkan nilai "Bagian" dari kolom BAGIAN
    bagian_value = current_df['BAGIAN'].iloc[0] if 'BAGIAN' in current_df.columns else 'Engine'
    return render_template('container_engine.html', bagian=bagian_value)

@app.route('/manalagi_deck')
def manalagi_deck():
    global current_df
    current_df = df3
    # Dapatkan nilai "Bagian" dari kolom BAGIAN
    bagian_value = current_df['BAGIAN'].iloc[0] if 'BAGIAN' in current_df.columns else 'Deck'
    return render_template('manalagi_deck.html', bagian=bagian_value)

@app.route('/manalagi_engine')
def manalagi_engine():
    global current_df
    current_df = df4  # Pastikan df4 sudah di-load dengan benar dan sesuai
    # Dapatkan nilai "Bagian" dari kolom BAGIAN
    bagian_value = current_df['BAGIAN'].iloc[0] if 'BAGIAN' in current_df.columns else 'Engine'
    # print(bagian_value)
    return render_template('manalagi_engine.html', bagian=bagian_value)

@app.route('/bc')
def bc():
    global current_df
    current_df = df5
    # Dapatkan nilai "Bagian" dari kolom BAGIAN
    bagian_value = current_df['BAGIAN'].iloc[0] if 'BAGIAN' in current_df.columns else 'BC'
    return render_template('bc.html', bagian=bagian_value)

@app.route('/mt')
def mt():
    global current_df
    current_df = df6
    # Dapatkan nilai "Bagian" dari kolom BAGIAN
    bagian_value = current_df['BAGIAN'].iloc[0] if 'BAGIAN' in current_df.columns else 'MT'
    return render_template('mt.html', bagian=bagian_value)

@app.route('/tb')
def tb():
    global current_df
    current_df = df8
    # Dapatkan nilai "Bagian" dari kolom BAGIAN
    bagian_value = current_df['BAGIAN'].iloc[0] if 'BAGIAN' in current_df.columns else 'TB'
    return render_template('tb.html', bagian=bagian_value)

@app.route('/tk')
def tk():
    global current_df
    current_df = df7
    # Dapatkan nilai "Bagian" dari kolom BAGIAN
    bagian_value = current_df['BAGIAN'].iloc[0] if 'BAGIAN' in current_df.columns else 'TK'
    return render_template('tk.html', bagian=bagian_value)

@app.route('/other')
def other():
    global current_df
    current_df = df9
    # Dapatkan nilai "Bagian" dari kolom BAGIAN
    bagian_value = current_df['BAGIAN'].iloc[0] if 'BAGIAN' in current_df.columns else 'Other'
    return render_template('other.html', bagian=bagian_value)

@app.route('/vessels-options')
def vessels_options():
    global current_df
    # Extract vessel names from the current DataFrame
    vessels_option = current_df['VESSEL'].unique().tolist() if 'VESSEL' in current_df.columns else []
    return jsonify({'vessels_option': vessels_option})

@app.route('/options', methods=['GET'])
def get_options():
    global current_df
    bagian_option = current_df["BAGIAN"].unique().tolist() if "BAGIAN" in current_df.columns else []
    cert_option = current_df["CERTIFICATE"].unique().tolist() if "CERTIFICATE" in current_df.columns else []
    rank_option = current_df["RANK"].unique().tolist() if "RANK" in current_df.columns else []
    vessel_option = current_df["VESSEL"].unique().tolist() if "VESSEL" in current_df.columns else []
    
    data = {
        "bagian_option": bagian_option,
        "cert_option": cert_option,
        "rank_option": rank_option,
        "vessel_option": vessel_option,
    }
    return jsonify(data)

@app.route('/get-recommendation', methods=['POST'])
def get_recommendation():
    global current_df
    data_candidate = request.json
    bagian = data_candidate["BAGIAN"]
    vessel_name = data_candidate["VESSEL"]
    rank = data_candidate["RANK"]
    certificate = data_candidate["CERTIFICATE"]
    age_range = (data_candidate["UMUR"], data_candidate["UMUR"])

    # Panggil getRecommendation dengan current_df sebagai parameter
    recommendations = getRecommendation(current_df, data_candidate, bagian, vessel_name, rank, certificate, age_range)
    result = recommendations.to_dict(orient="records")
    return jsonify(result)

@app.route('/get-manual-search', methods=['POST'])
def get_manual_search():
    global current_df
    data_candidate = request.json
    bagian = data_candidate["BAGIAN"]
    vessel_name = data_candidate["VESSEL"]
    age_range = (data_candidate["LB"], data_candidate["UB"])

    # Call search_candidate with current_df as the parameter
    filtered_candidates = search_candidate(current_df, bagian, vessel_name, age_range)
    
    if filtered_candidates.empty:
        return jsonify([])

    recommendations = getRecommendation(current_df, data_candidate, bagian, vessel_name, data_candidate["RANK"], data_candidate["CERTIFICATE"], age_range)
    
    # Ensure PHONE1, PHONE2, PHONE3, and PHONE4 are included in the response
    result = recommendations[['SEAMAN CODE', 'SEAFARER CODE', 'SEAMAN NAME', 'RANK', 'VESSEL', 'UMUR', 'CERTIFICATE', 'PHONE1', 'PHONE2', 'PHONE3', 'PHONE4', 'DAY REMAINS']].to_dict(orient="records")
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
