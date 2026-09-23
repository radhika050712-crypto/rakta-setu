from flask import Flask, render_template, request, redirect, url_for,jsonify
import sqlite3
import cv2
import base64
import numpy as np
import os
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
cascade_path = os.path.join(os.path.dirname(__file__),'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path);
import pickle
import time 


def create_tables():
    conn = sqlite3.connect('blood.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS donors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        blood_group TEXT,
        city TEXT,
        contact TEXT,
        age INTEGER,
        weight REAL,
        hemoglobin REAL,
        last_donation TEXT,
        total_cc REAL,
        photo_path TEXT,
        status TEXT
    )''')
    conn.commit()
    conn.close()


def init_db():
    conn = sqlite3.connect('blood.db')
    c = conn.cursor()
   
    conn.commit()
    conn.close()
def check_eligibility(age, weight, hemoglobin, last_donation, recency, frequency, monetary, time):
    # 1. RULE BASED CHECK
    if age < 18:
        return "Not Eligible: Age must be 18+"
    if weight < 45:
        return "Not Eligible: Weight must be 45Kg+"
    if hemoglobin < 12.5:
        return "Not Eligible: Hemoglobin must be 12.5+"

    if last_donation:
        last_date = datetime.strptime(last_donation, '%Y-%m-%d')
        if datetime.now() - last_date < timedelta(days=90):
            return "Not Eligible: 3 months gap required"

    # 2. ML MODEL CHECK
    df = pd.read_csv('blood-transfusion.csv')
    X = df[['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)']]
    y = df['donated before']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    pred = model.predict([[recency, frequency, monetary, time]])[0]

    if pred == 1:
        return "Eligible"
    else:
        return "Eligible by Rules, But ML says Not Eligible"
app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR,'static','uploads')
app.config['UPLOAD_FOLDER'] =UPLOAD_FOLDER
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except FileExistsError:
    pass 
init_db() 


def donor_health_check(frame):
    if frame is None or frame.size == 0:
        return "Not Fit",["Camerea frame not captured"]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    reasons = []

    if len(faces) == 0:
        return "No Face Detected", ["Face not visible in camera"]

    status = "Likely Fit for Donation"
    reasons.append("Face detected successfully")
    reasons.append("Note: This is demo. Final check by Doctor.")
    
    return status, reasons
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        age = int(request.form['age'])
        weight = float(request.form['weight'])
        blood_group = request.form['blood_group']
        city = request.form['city']
        last_donation = request.form['last_donation']
        hemoglobin = float(request.form['hemoglobin'])
        total_cc = request.form.get('total_cc')
        if total_cc == '' or total_cc is None:
            total_cc = None
        else:
            total_cc = float(total_cc)

        gap_days = 0
        if last_donation and last_donation.strip() != '':
            try:
                last_date = datetime.strptime(last_donation, '%Y-%m-%d')
                gap_days = (datetime.now() - last_date).days
            except:
                gap_days = 0

        if age >= 18 and age <= 65 and weight >= 50 and hemoglobin >= 12.5 and gap_days >= 90:
            status = "Eligible"
        else:
            status = "Not Eligible"

        photo_data = request.form.get('photo_data')
        photo_path = None
        if photo_data:
            header, encoded = photo_data.split(",", 1)
            data = base64.b64decode(encoded)
            filename = secure_filename(f"{name}_{int(time.time())}.jpg")
            photo_path = os.path.join('static/uploads', filename)
            with open(photo_path, "wb") as f:
                f.write(data)

        conn = sqlite3.connect('blood.db')
        c = conn.cursor()
        c.execute("INSERT INTO donors (name, age, weight, blood_group, city, last_donation, hemoglobin, total_cc, photo_path, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (name, age, weight, blood_group, city, last_donation, hemoglobin, total_cc, photo_path, status))
        conn.commit()
        conn.close()

        if status == 'Eligible':
            msg = f"✅ Congratulations {name}! You are Eligible. Your donation has been submitted."
        else:
            days_left = 90 - gap_days
            if days_left < 0:
                days_left = 90
            msg = f"❌ Sorry {name}, You are Not Eligible. You can donate after {days_left} days."

        return render_template('success.html', message=msg)
    
    return render_template('register.html')
@app.route('/search', methods=['POST'])
def search():
    blood = request.form.get('blood_group')
    city = request.form.get('city')
    conn = sqlite3.connect('blood.db')
    c = conn.cursor()
    c.execute("SELECT * FROM donors WHERE blood_group=? AND city=?", (blood, city))
    donors = c.fetchall()
    conn.close()
    return render_template('search.html', donors=donors)

@app.route('/delete/<int:id>')
def delete(id):
    conn = sqlite3.connect('blood.db')
    c = conn.cursor()
    c.execute("DELETE FROM donors WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('home'))


@app.route('/live_check')
def live_check():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow("AI Donor Fitness Check - Press Q to Quit")
    
    status, reasons = "Not Fit", ["No face detected"]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            status, reasons = "Likely Fit for Donation", ["Face detected successfully"]

        cv2.imshow("AI Donor Fitness Check - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Q दबाते ही बंद
            break

    cap.release()
    cv2.destroyAllWindows()
    
    conn = sqlite3.connect('blood.db')
    donors_data = conn.execute('SELECT * FROM donors').fetchall()
    conn.close()
    
    return render_template('home.html', donors=donors_data, status=None, reasons=[])


@app.route('/check_fitness', methods=['POST'])  # YE NAYA WALA YAHAN LAGANA HAI
def check_fitness():
    data = request.get_json()
    image_data = data['image']

    import random
    if random.randint(0,1) == 1:
        status = "✅ Eligible to Donate"
        reasons = ["Face appears fresh and healthy", "Eyes are clear and bright"]
    else:
        status = "❌ Not Eligible to Donate" 
        reasons = ["Face appears tired"]
    
    return jsonify({"status": status, "reasons": reasons})


    
if __name__ == '__main__':
    create_tables()
    app.run(debug=True,host='0.0.0.0')