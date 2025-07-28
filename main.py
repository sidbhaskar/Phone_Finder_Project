from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Global variables to store preprocessed data and models
dataset = None          
dataset_new = None      
similarity = None       
ps = PorterStemmer()    
display_df = None       
cv = None               
vectors = None          

# Global wishlist (one-time use, not per user/session)
wishlist = []

def load_and_preprocess_data():
    global dataset, dataset_new, similarity, display_df, cv, vectors

    csv_path = 'smart_phones1.csv'
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found. Please ensure the CSV file is in the same directory as main.py.")
        dataset = pd.DataFrame(columns=['Phone_Name', 'Reviews', 'RAM', 'ROM', 'Display_Size', 'Processor', 'Battery', 'Rear_Camera', 'Front_Camera', 'Current_Price', 'Original_Price', 'image'])
        dataset_new = pd.DataFrame(columns=['Phone_Name', 'details'])
        similarity = np.array([])
        display_df = pd.DataFrame(columns=['Phone_Name', 'ROM', 'Current_Price', 'image', 'Processor', 'RAM'])
        cv = CountVectorizer()
        vectors = np.array([])
        return

    raw_data = pd.read_csv(csv_path)
    dataset = raw_data.copy()

    display_df = pd.DataFrame()
    display_df['Phone_Name'] = raw_data['Phone_Name']
    display_df['Processor'] = raw_data['Processor']
    display_df['image'] = raw_data['image'] if 'image' in raw_data.columns else ''
    display_df['Colour'] = raw_data['Phone_Name'].str.extract(r'\(([^,]+),').fillna('N/A')
    display_df['Current_Price'] = raw_data['Current_Price'].str.replace('₹', '').str.replace(',', '')
    display_df['Current_Price'] = pd.to_numeric(display_df['Current_Price'], errors='coerce')
    display_df['Current_Price'] = display_df['Current_Price'].fillna(display_df['Current_Price'].median() if not display_df['Current_Price'].isnull().all() else 0)
    display_df['Current_Price_Formatted'] = display_df['Current_Price'].apply(lambda x: f'₹{int(x)}' if pd.notna(x) and str(x).replace('.', '').isdigit() else 'N/A')

    # Extract numeric RAM and ROM for filtering
    display_df['RAM'] = raw_data['RAM'].astype(str).str.extract(r'(\d+)').astype(float)
    rom_pattern_display = r'(\d+\.?\d*)\s*(GB|MB)'
    extracted_rom_display = raw_data['ROM'].astype(str).str.extract(rom_pattern_display, expand=True)
    def format_rom_for_display(row_raw_rom, extracted_num, extracted_unit):
        if pd.notna(extracted_num) and pd.notna(extracted_unit):
            return f"{int(float(extracted_num))} {extracted_unit}"
        elif pd.notna(row_raw_rom) and ('GB' in str(row_raw_rom) or 'MB' in str(row_raw_rom)):
            return str(row_raw_rom).strip()
        else:
            return 'N/A'
    display_df['ROM_Formatted'] = raw_data.apply(lambda row: format_rom_for_display(row['ROM'], extracted_rom_display.loc[row.name, 0], extracted_rom_display.loc[row.name, 1]), axis=1)
    # Numeric ROM in GB for filtering
    display_df['ROM_GB'] = extracted_rom_display[0].astype(float)
    display_df['ROM_GB'] = display_df['ROM_GB'].where(extracted_rom_display[1] == 'GB', display_df['ROM_GB'] / 1024)

def search_phones_by_brand_and_budget(brand, max_budget, min_ram_gb=None, min_rom_gb=None, top_n=20):
    global display_df

    filtered = display_df.copy()
    if brand:
        filtered = filtered[filtered['Phone_Name'].str.lower().str.contains(brand)]
    if max_budget:
        try:
            max_budget_int = int(max_budget)
            filtered = filtered[filtered['Current_Price'] <= max_budget_int]
        except Exception:
            pass
    if min_ram_gb:
        try:
            min_ram_gb_val = float(min_ram_gb)
            filtered = filtered[filtered['RAM'] >= min_ram_gb_val]
        except Exception:
            pass
    if min_rom_gb:
        try:
            min_rom_gb_val = float(min_rom_gb)
            filtered = filtered[filtered['ROM_GB'] >= min_rom_gb_val]
        except Exception:
            pass

    results_list = []
    for _, row in filtered.head(top_n).iterrows():
        results_list.append({
            'Phone': row['Phone_Name'],
            'Processor': row['Processor'],
            'RAM': int(row['RAM']) if pd.notna(row['RAM']) else 'N/A',
            'ROM': row['ROM_Formatted'] if 'ROM_Formatted' in row else row['ROM'],
            'Price': row['Current_Price_Formatted'] if 'Current_Price_Formatted' in row else f"₹{row['Current_Price']}" if pd.notna(row['Current_Price']) else 'N/A',
            'ImageURL': row['image'] if 'image' in row and pd.notna(row['image']) and row['image'] else ''
        })
    return results_list

# Initialize data on app startup
with app.app_context():
    load_and_preprocess_data()

@app.route('/')
def index():
    return render_template('index.html', search_results=[], brand='', max_budget='', min_ram='', min_rom='', wishlist=wishlist)

@app.route('/search')
def search():
    brand = request.args.get('brand', '').strip().lower()
    max_budget = request.args.get('max_budget', '').strip()
    min_ram = request.args.get('min_ram', '').strip()
    min_rom = request.args.get('min_rom', '').strip()
    results = []

    if brand or max_budget or min_ram or min_rom:
        results = search_phones_by_brand_and_budget(brand, max_budget, min_ram, min_rom)
    return render_template('index.html', search_results=results, brand=brand, max_budget=max_budget, min_ram=min_ram, min_rom=min_rom, wishlist=wishlist)

@app.route('/add_to_wishlist', methods=['POST'])
def add_to_wishlist():
    data = request.json
    phone = data.get('phone')
    imageURL = data.get('imageURL')
    if not phone:
        return jsonify({'success': False}), 400
    # Avoid duplicates
    for item in wishlist:
        if item['phone'] == phone:
            return jsonify({'success': True})
    wishlist.append({'phone': phone, 'imageURL': imageURL})
    return jsonify({'success': True})

@app.route('/remove_from_wishlist', methods=['POST'])
def remove_from_wishlist():
    data = request.json
    phone = data.get('phone')
    if not phone:
        return jsonify({'success': False}), 400
    global wishlist
    wishlist = [item for item in wishlist if item['phone'] != phone]
    return jsonify({'success': True})

@app.route('/get_wishlist')
def get_wishlist():
    return jsonify({'wishlist': wishlist})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')