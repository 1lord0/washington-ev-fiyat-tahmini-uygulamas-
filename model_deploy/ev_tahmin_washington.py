import os
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Veriyi oku
url="model_deploy/yenidata1.csv"
df = pd.read_csv(url)

# Sadece belirli şehirleri seç (örneğin: Shoreline, Seattle, Renton, Woodinville)
allowed_cities = ['Shoreline', 'Seattle', 'Renton', 'Woodinville','Federal Way','Kirkland','Issaquah'    ]
df = df[df["city"].isin(allowed_cities)]

# Her şehir için en iyi parametreler (önceden belirlenmiş)
best_params_city = {
    'Shoreline': {'alpha': 0.7742636826811278, 'l1_ratio': 0.8888888888888888},
    'Seattle': {'alpha': 0.0001, 'l1_ratio': 1.0},
    'Renton': {'alpha': 0.7742636826811278, 'l1_ratio': 0.1111111111111111},
    'Woodinville':{'alpha': 10.0, 'l1_ratio': 0.0},
     'Federal Way': {'alpha': 0.05994842503189409, 'l1_ratio': 0.3333333333333333},
     'Kirkland': {'alpha': 0.0001, 'l1_ratio': 1.0},
    'Issaquah': {'alpha': 0.0001, 'l1_ratio': 1.0}
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
    "Woodinville": "model_deploy/woodinville_model.pkl",
    "Federal Way": "model_deploy/federal_way_model.pkl",
    "Kirkland":"model_deploy/kirkland_model.pkl",
    "Issaquah":'model_deploy/issaquah_model.pkl'
}

st.title("🏠 Ev Fiyat Tahmin Uygulaması")
sehir = st.sidebar.selectbox("Şehir Seçiniz", list(model_paths.keys()))

st.write("Elastic Net modeli kullanarak ev fiyatlarını tahmin edin.")

# Kullanıcı girdilerini alıyoruz.
st.sidebar.header("Ev Özelliklerini Girin")
bedrooms = st.sidebar.number_input("Yatak Odası Sayısı", min_value=0, max_value=10, value=3, step=1)
bathrooms = st.sidebar.number_input("Banyo Sayısı", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
sqft_living = st.sidebar.number_input("Yaşam Alanı (sqft)", min_value=100, max_value=10000, value=2000)
sqft_lot = st.sidebar.number_input("Arsa Alanı (sqft)", min_value=500, max_value=50000, value=5000)
floors = st.sidebar.number_input("Kat Sayısı", min_value=1, max_value=5, value=1)
waterfront = st.sidebar.selectbox("Su Kenarı mı?", options=["Hayır", "Evet"])
# Binary dönüşüm: "Evet" için 1, "Hayır" için 0
waterfront = 1 if waterfront == "Evet" else 0

view = st.sidebar.number_input("Manzara Puanı", min_value=0, max_value=5, value=0, step=1)
condition = st.sidebar.number_input("Ev Durumu (1-5)", min_value=1, max_value=5, value=3, step=1)
sqft_above = st.sidebar.number_input("Üst Kat Alanı (sqft)", min_value=100, max_value=10000, value=1500)
sqft_basement = st.sidebar.number_input("Bodrum Alanı (sqft)", min_value=0, max_value=5000, value=500)
ev_yenilendi_mi = st.sidebar.selectbox("Ev Yenilendi mi?", options=["Hayır", "Evet"])
# Binary dönüşüm: "Evet" için 1, "Hayır" için 0
ev_yenilendi_mi = 1 if ev_yenilendi_mi == "Evet" else 0
ev_yili = st.sidebar.number_input("Ev Yapım Yılı", min_value=1800, max_value=2025, value=2000, step=1)

# "Tahmin Yap" butonuna basıldığında tahmin işlemi başlıyor.
if st.sidebar.button("Tahmin Yap"):
    model_path = model_paths[sehir]
    if os.path.exists(model_path):
        # Modeli yükle
        model = joblib.load(model_path)
        # Girdi verilerini modele uygun formata getiriyoruz.
        input_data = np.array([[bedrooms, bathrooms,sqft_living,sqft_lot,floors ,waterfront,view,condition,sqft_above,sqft_basement,ev_yenilendi_mi,ev_yili]])

        # Tahmin yap
        tahmin = model.predict(input_data)
        st.success(f"{sehir} için tahmini ev fiyatı: {tahmin[0]:.2f} dolar")
    else:
        st.error(f"{sehir} modeli bulunamadı. Lütfen model dosyasını kontrol edin.")
