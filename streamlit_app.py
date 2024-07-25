import streamlit as st
from flask import Flask, request, jsonify
from joblib import load
import logging
import requests
import json

# Inisialisasi logging
logging.basicConfig(level=logging.DEBUG)

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Fungsi untuk memuat model
def load_model():
    global DTReg
    try:
        DTReg = load('windprediction_DTReg.pkl')
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error("Error loading model: %s", e)
        DTReg = None

# Muat model saat aplikasi dijalankan
load_model()

# Endpoint untuk memprediksi
@app.route('/predict', methods=['POST'])
def predict():
    if DTReg is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Ambil data dari request POST
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        Tavg = data['Tavg']
        RH_avg = data['RH_avg']
        
        # Lakukan prediksi dengan model
        prediction = DTReg.predict([[Tavg, RH_avg]])
        result = {'ff_x': prediction[0]}
        
        logging.debug(f"Sending result: {result}")
        return jsonify(result)
    except Exception as e:
        logging.error("Prediction error: %s", e)
        return jsonify({'error': str(e)}), 500

# Streamlit interface
st.title('Wind Prediction')

# Input dari pengguna untuk temperatur dan kelembaban relatif rata-rata
Tavg = st.number_input('Average Temperature (Tavg)')
RH_avg = st.number_input('Average Relative Humidity (RH_avg)')

# Tombol untuk melakukan prediksi
if st.button('Predict'):
    # Buat data untuk dikirim ke server Flask
    data = {'Tavg': Tavg, 'RH_avg': RH_avg}
    
    # URL endpoint prediksi
    url = 'https://wind-prediction-anemoi.streamlit.app/'  # Sesuaikan dengan URL server Flask
    
    # Headers untuk request POST
    headers = {'Content-Type': 'application/json'}
    
    try:
        # Kirim request POST ke server Flask
        response = requests.post(url, data=json.dumps(data), headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses
        
        # Handle respons dari server Flask
        result = response.json()
        if 'ff_x' in result:
            st.success(f"Prediction: {result['ff_x']}")
        else:
            st.error('Error in prediction response')
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error during request to Flask server: {e}")
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response from Flask server: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Jalankan aplikasi Streamlit sebagai server Flask
app.run(port=80)
