import os
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Veriyi oku
df = pd.read_csv("yenidata1.csv")

# Sadece belirli şehirleri seç (örneğin: Shoreline, Seattle, Renton, Woodinville)
allowed_cities = ['Shoreline', 'Seattle', 'Renton', 'Woodinville']
df = df[df["city"].isin(allowed_cities)]

# Her şehir için en iyi parametreler (önceden belirlenmiş)
best_params_city = {
    'Shoreline': {'alpha': 0.7742636826811278, 'l1_ratio': 0.8888888888888888},
    'Seattle': {'alpha': 0.0001, 'l1_ratio': 1.0},
    'Renton': {'alpha': 0.7742636826811278, 'l1_ratio': 0.1111111111111111},
    'Woodinville': {'alpha': 10.0, 'l1_ratio': 0.0}
}

# Model dosyalarının kaydedileceği klasörü oluştur (eğer yoksa)
model_dir = "model_deploy"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Her şehir için modeli eğit ve kaydet
for city in allowed_cities:
    df_city = df[df["city"] == city]
    
    # Özellikler ve hedef değişken belirle
    X = df_city.drop(columns=['price', 'city'])
    y = df_city["price"]
    
    # Eğitim ve test verisini ayır (örneğin %80 eğitim, %20 test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Şehir için en iyi parametreleri al
    best_params = best_params_city[city]
    
    # Elastic Net modeli oluştur ve eğit
    model = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'])
    model.fit(X_train, y_train)
    
    # Model performansını değerlendir
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{city}: MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")
    
    # Modeli kaydet
    model_path = os.path.join(model_dir, f"{city.lower()}_model.pkl")
    joblib.dump(model, model_path)
    print(f"{city} modeli kaydedildi: {model_path}")


import streamlit as st
import joblib
import numpy as np
import os

# Model dosya yollarını tanımlıyoruz.
# Bu modelleri daha önceden Elastic Net ile eğitip joblib.dump ile kaydetmiş olmanız gerekiyor.
model_paths = {
    "Shoreline": "model_deploy/shoreline_model.pkl",
    "Seattle": "model_deploy/seattle_model.pkl",
    "Renton": "model_deploy/renton_model.pkl",
    "Woodinville": "model_deploy/woodinville_model.pkl"
}

st.title("🏠 Ev Fiyat Tahmin Uygulaması")
st.write("Elastic Net modeli kullanarak ev fiyatlarını tahmin edin.")

# Kullanıcı girdilerini alıyoruz.
st.sidebar.header("Ev Özelliklerini Girin")
sehir = st.sidebar.selectbox("Şehir Seçiniz", list(model_paths.keys()))
metrekare = st.sidebar.number_input("Metrekare (m²)", min_value=50, max_value=500, value=100)
oda_sayisi = st.sidebar.number_input("Oda Sayısı", min_value=1, max_value=10, value=3)
bina_yasi = st.sidebar.number_input("Bina Yaşı", min_value=0, max_value=100, value=10)

# "Tahmin Yap" butonuna basıldığında tahmin işlemi başlıyor.
if st.sidebar.button("Tahmin Yap"):
    model_path = model_paths[sehir]
    if os.path.exists(model_path):
        # Modeli yükle
        model = joblib.load(model_path)
        # Girdi verilerini modele uygun formata getiriyoruz.
        input_data = np.array([[metrekare, oda_sayisi, bina_yasi]])
        # Tahmin yap
        tahmin = model.predict(input_data)
        st.success(f"{sehir} için tahmini ev fiyatı: {tahmin[0]:.2f} TL")
    else:
        st.error(f"{sehir} modeli bulunamadı. Lütfen model dosyasını kontrol edin.")
