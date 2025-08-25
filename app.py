import joblib
from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import uuid
import io
import openpyxl

app = Flask(__name__, template_folder='.')

# --- Muat Pipeline Model dan Daftar Fitur ---
pipeline_path = 'logistic_regression_penumpang_pipeline.pkl'
features_with_moda_path = 'model_features_with_moda.pkl'
numerical_features_path = 'numerical_features.pkl'
categorical_features_path = 'categorical_features.pkl'

model_pipeline = None
model_features_with_moda = None
numerical_features = None
categorical_features = None

try:
    model_pipeline = joblib.load(pipeline_path)
    model_features_with_moda = joblib.load(features_with_moda_path)
    numerical_features = joblib.load(numerical_features_path)
    categorical_features = joblib.load(categorical_features_path)
    print("Pipeline Model dan Daftar Fitur berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat pipeline/fitur: {e}")
    exit()

# --- Simulasi Database In-Memory (Global List) ---
db_data = []
moda_medians = {}

# --- Utility Function to Calculate Medians and Assign Crowd Level ---
# --- MODIFICATION: This function is correct, but will now work properly with the clean data ---
def update_crowd_levels():
    global db_data, moda_medians
    if not db_data:
        moda_medians = {}
        return

    df = pd.DataFrame(db_data)
    df['tanggal_dt'] = pd.to_datetime(df['tanggal'])

    # Calculate median for each mode ('krl' and 'transjakarta' separately)
    moda_medians = df.groupby('jenis_moda')['jumlah_penumpang_per_hari'].median().to_dict()

    # Assign 'tingkat_keramaian'
    for record in db_data:
        mode = record['jenis_moda']
        jumlah_penumpang = record['jumlah_penumpang_per_hari']
        median_for_mode = moda_medians.get(mode)

        if median_for_mode is not None:
            if jumlah_penumpang > median_for_mode:
                record['tingkat_keramaian'] = 'TINGGI'
            else:
                record['tingkat_keramaian'] = 'RENDAH'
        else:
            record['tingkat_keramaian'] = 'N/A'
            
    db_data.sort(key=lambda x: (x['jenis_moda'], x['tanggal']))

# --- MODIFICATION 1: Update Initial Data Loading Logic ---
# This block now filters for KRL/Transjakarta and standardizes to lowercase internally.
# --- Inisialisasi db_data dengan Data CSV saat Startup ---
historical_data_path = 'Jumlah_Penumpang_Angkutan_Umum_yang_Terlayani_Perhari.csv'
try:
    initial_historical_df = pd.read_csv(historical_data_path, delimiter=';')

    # Data cleaning
    initial_historical_df['jumlah_penumpang_per_hari'] = pd.to_numeric(
        initial_historical_df['jumlah_penumpang_per_hari'].astype(str).str.replace(',', '.'), errors='coerce'
    )
    initial_historical_df['tanggal_dt'] = pd.to_datetime(initial_historical_df['tanggal'], format='%d/%m/%Y', errors='coerce')
    initial_historical_df.dropna(subset=['tanggal_dt', 'jumlah_penumpang_per_hari', 'jenis_moda'], inplace=True)

    # Standardize to lowercase to match the training script
    initial_historical_df['jenis_moda'] = initial_historical_df['jenis_moda'].str.lower()
    
    # Filter for 'krl' and 'transjakarta' only
    modes_to_keep = ['krl', 'transjakarta']
    initial_historical_df = initial_historical_df[initial_historical_df['jenis_moda'].isin(modes_to_keep)]

    initial_historical_df = initial_historical_df.sort_values(by=['jenis_moda', 'tanggal_dt']).reset_index(drop=True)

    for index, row in initial_historical_df.iterrows():
        db_data.append({
            'id': str(uuid.uuid4()),
            'tanggal': row['tanggal_dt'].strftime('%Y-%m-%d'),
            'jenis_moda': row['jenis_moda'],  # Stored as 'krl' or 'transjakarta'
            'jumlah_penumpang_per_hari': row['jumlah_penumpang_per_hari']
        })
        
    print(f"Database diisi dengan {len(db_data)} record (hanya krl dan transjakarta).")
    update_crowd_levels()
    print("Tingkat keramaian awal dihitung.")
except Exception as e:
    print(f"Error saat menginisialisasi data: {e}")


# --- Rute Flask ---
@app.route('/')
def home():
    return render_template('index.html')

# --- MODIFICATION 2: Update Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_date_str = data.get('input_date')
        prediction_days = int(data.get('prediction_days', 7))

        if not input_date_str:
            return jsonify({'error': "Missing 'input_date' in input data."}), 400

        start_date = pd.to_datetime(input_date_str)
        all_predictions = {}
        
        live_data_for_features = pd.DataFrame(db_data)
        if not live_data_for_features.empty:
            live_data_for_features['jumlah_penumpang_per_hari'] = pd.to_numeric(live_data_for_features['jumlah_penumpang_per_hari'], errors='coerce')
            live_data_for_features['tanggal'] = pd.to_datetime(live_data_for_features['tanggal'], errors='coerce')
            live_data_for_features.dropna(subset=['jumlah_penumpang_per_hari', 'tanggal', 'jenis_moda'], inplace=True)
            live_data_for_features = live_data_for_features.sort_values(by=['jenis_moda', 'tanggal']).reset_index(drop=True)

        for i in range(prediction_days):
            current_date = start_date + timedelta(days=i)
            current_date_str = current_date.strftime('%Y-%m-%d')
            predictions_for_day = {}

            # Use lowercase for prediction to match the model
            modes_to_predict = ['krl', 'transjakarta']

            for mode in modes_to_predict:
                numeric_input_dict = {
                    'Tahun': current_date.year, 'Hari_dalam_Bulan': current_date.day,
                    'Hari_sin': np.sin(2 * np.pi * current_date.dayofweek / 7),
                    'Hari_cos': np.cos(2 * np.pi * current_date.dayofweek / 7),
                    'Bulan_sin': np.sin(2 * np.pi * current_date.month / 12),
                    'Bulan_cos': np.cos(2 * np.pi * current_date.month / 12),
                    'is_weekend': 1 if current_date.dayofweek >= 5 else 0
                }

                weekly_avg, lag_1, lag_7 = 0, 0, 0
                if not live_data_for_features.empty:
                    mode_live_data = live_data_for_features[live_data_for_features['jenis_moda'] == mode]
                    
                    relevant_weekly_data = mode_live_data[
                        (mode_live_data['tanggal'].dt.dayofweek == current_date.dayofweek) &
                        (mode_live_data['tanggal'] < current_date) &
                        (mode_live_data['tanggal'] >= current_date - pd.Timedelta(weeks=4))
                    ]
                    if not relevant_weekly_data.empty:
                        weekly_avg = relevant_weekly_data['jumlah_penumpang_per_hari'].mean()

                    prev_day_data = mode_live_data[mode_live_data['tanggal'] == current_date - pd.Timedelta(days=1)]
                    if not prev_day_data.empty:
                        lag_1 = prev_day_data['jumlah_penumpang_per_hari'].iloc[0]

                    prev_week_data = mode_live_data[mode_live_data['tanggal'] == current_date - pd.Timedelta(days=7)]
                    if not prev_week_data.empty:
                        lag_7 = prev_week_data['jumlah_penumpang_per_hari'].iloc[0]

                numeric_input_dict['Rata_rata_penumpang_3_minggu_lalu'] = weekly_avg
                numeric_input_dict['lag_1_hari'] = lag_1
                numeric_input_dict['lag_7_hari'] = lag_7

                input_data_df = pd.DataFrame([numeric_input_dict])
                input_data_df[categorical_features[0]] = mode
                input_data_df = input_data_df[model_features_with_moda]

                prediction = model_pipeline.predict(input_data_df)
                prediction_proba = model_pipeline.predict_proba(input_data_df)

                result_label = 'Jumlah Penumpang TINGGI' if prediction[0] == 1 else 'Jumlah Penumpang RENDAH'
                probability_high = prediction_proba[0][1]

                # Use capitalized key for the final JSON to match frontend table
                display_mode = 'KRL' if mode == 'krl' else 'Transjakarta'
                predictions_for_day[display_mode] = {'prediction': result_label, 'probability_high': float(probability_high)}
                
            all_predictions[current_date_str] = predictions_for_day

        return jsonify(all_predictions)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

# --- CRUD Endpoints ---
# --- MODIFICATION 3: Update GET /data to capitalize for display ---
@app.route('/data', methods=['GET'])
def get_all_data():
    display_data = [record.copy() for record in db_data]
    for record in display_data:
        record['jenis_moda'] = record['jenis_moda'].upper() if record['jenis_moda'] == 'krl' else record['jenis_moda'].capitalize()
    return jsonify(display_data)

# --- MODIFICATION 4: Update POST /data to save as lowercase ---
@app.route('/data', methods=['POST'])
def add_data():
    global db_data
    try:
        new_record = request.get_json(force=True)
        if not all(k in new_record for k in ['tanggal', 'jenis_moda', 'jumlah_penumpang_per_hari']):
            return jsonify({'error': 'Missing data fields'}), 400
        
        datetime.strptime(new_record['tanggal'], '%Y-%m-%d')
        
        if new_record['jenis_moda'].upper() not in ['KRL', 'TRANSJAKARTA']:
            return jsonify({'error': 'Invalid jenis_moda. Must be KRL or Transjakarta.'}), 400

        record_to_add = {
            'id': str(uuid.uuid4()),
            'tanggal': new_record['tanggal'],
            'jenis_moda': new_record['jenis_moda'].lower(), # Store as lowercase
            'jumlah_penumpang_per_hari': new_record['jumlah_penumpang_per_hari']
        }
        db_data.append(record_to_add)
        update_crowd_levels()
        return jsonify(record_to_add), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- MODIFICATION 5: Update PUT /data/<id> to save as lowercase ---
@app.route('/data/<id>', methods=['PUT'])
def update_data(id):
    global db_data
    try:
        updated_fields = request.get_json(force=True)
        record_found = False
        for record in db_data:
            if record['id'] == id:
                if 'tanggal' in updated_fields:
                    record['tanggal'] = updated_fields['tanggal']
                if 'jenis_moda' in updated_fields:
                    # Save updated mode as lowercase
                    record['jenis_moda'] = updated_fields['jenis_moda'].lower()
                if 'jumlah_penumpang_per_hari' in updated_fields:
                    record['jumlah_penumpang_per_hari'] = updated_fields['jumlah_penumpang_per_hari']
                
                record_found = True
                update_crowd_levels()
                return jsonify(record), 200
        if not record_found:
            return jsonify({'error': 'Record not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data/<id>', methods=['DELETE'])
def delete_data(id):
    global db_data
    initial_len = len(db_data)
    db_data = [record for record in db_data if record['id'] != id]
    if len(db_data) < initial_len:
        update_crowd_levels()
        return jsonify({'message': 'Record deleted successfully'}), 200
    else:
        return jsonify({'error': 'Record not found'}), 404

# --- MODIFICATION 6: Update Excel Export to capitalize for display ---
@app.route('/export_excel', methods=['GET'])
def export_excel():
    try:
        if not db_data:
            return jsonify({'error': 'No data to export'}), 404

        df_export = pd.DataFrame(db_data)
        
        # Capitalize for user-friendly export
        df_export['jenis_moda'] = df_export['jenis_moda'].apply(lambda x: x.upper() if x == 'krl' else x.capitalize())

        export_columns = ['tanggal', 'jenis_moda', 'jumlah_penumpang_per_hari', 'tingkat_keramaian']
        df_export = df_export[export_columns]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Data Penumpang')
        output.seek(0)

        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            download_name='data_penumpang_historis.xlsx',
            as_attachment=True
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Error exporting Excel file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)