


import pandas as pd

import numpy as np


df = pd.read_csv('model_deploy/data.csv')  # dosya projenin içinde olmalı


import sys
df.drop(["date","country","statezip","street"],axis=1,inplace=True)


df[df["city"]=="Kent"]
import numpy as np
df["ev_yenilendi_mi"] = df["yr_renovated"].apply(lambda x: 1 if x > 0 else 0)
city_counts = df['city'].value_counts()
  # Şehirdeki veri sayısını hesapla
low_population_cities = city_counts[city_counts < 100].index  # 100'den az olanları belirle
df = df[~df['city'].isin(low_population_cities)]  # Bu şehirleri çıkar

print(f"✅ Güncellenmiş veri setinde {df['city'].nunique()} farklı şehir kaldı.")


df["ev_yili"]=df.apply(lambda row: row["yr_built"] if row["yr_renovated"]==0  else row["yr_renovated"],axis=1)


df=df.drop(["yr_built","yr_renovated"],axis=1)




# 'price' sütunundaki 'e' karakterlerini sayısal değere çevirme
df['price'] = df['price'].apply(lambda x: int(float(x)))

df=df[~(df["city"]=="Kent")]

from sklearn.model_selection import train_test_split
#gridsearch ile hangi şehre hangi parametreleri seçmeliyiz bunu bir sözlüğe aktarıyoruz
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet



df = df[~df['city'].isin(df['city'].value_counts()[df['city'].value_counts() < 100].index)]
best_param_city=[]
best_params_city = {
    'Shoreline': {'alpha': 0.7742636826811278, 'l1_ratio': 0.8888888888888888},
    'Seattle': {'alpha': 0.0001, 'l1_ratio': 1.0},
    'Renton': {'alpha': 0.7742636826811278, 'l1_ratio': 0.1111111111111111},
    'Woodinville': {'alpha': 10.0, 'l1_ratio': 0.0}
}

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


predictions_by_city = {}



for city in df['city'].unique():
    df_city = df[df['city'] == city]
    n_samples = len(df_city)

    print(f"\nŞehir: {city} için model eğitiliyor...")

    X = df_city.drop(columns=['price', 'city'])  
    y = df_city['price']  

    if n_samples < 10:
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # En iyi parametreleri al
    best_params = best_params_by_city[city]

    # Modeli en iyi parametrelerle eğit
    model = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'])
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    # Hata hesapla
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  # R^2 değeri hesaplama
    
    # Sonuçları sakla
    predictions_by_city[city] = {
        'Gerçek Değerler': y_test.values,
        'Tahminler': y_pred,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }

    print(f"{city} için eğitim tamamlandı. MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")



import pickle

import streamlit as st
import joblib
import numpy as np
import pandas as pd

import os




import streamlit as st
import joblib
import numpy as np

# Model dosya yolları
model_paths = {
    "Seattle": 'model_deploy/seattle_model.pkl',
    "Renton": 'model_deploy/renton_model.pkl',
    "Bellevue": 'C:\\Users\\eren\\Desktop\\model_deploy\\bellevue_model.pkl',
    "Shoreline": 'model_deploy/shoreline_model.pkl',
    "Woodinville": 'model_deploy/woodinville_model.pkl'
}

# Streamlit başlık ve açıklama
st.title("🏠 Ev Fiyat Tahmin Uygulaması")
st.write("Bu uygulama, Elastic Net modeli kullanarak ev fiyatlarını tahmin eder.")

# Kullanıcıdan girdiler alma
st.sidebar.header("Ev Özelliklerini Girin")

metrekare = st.sidebar.number_input("Metrekare (m²)", min_value=50, max_value=500, value=100)
oda_sayisi = st.sidebar.number_input("Oda Sayısı", min_value=1, max_value=10, value=3)
bina_yasi = st.sidebar.number_input("Bina Yaşı", min_value=0, max_value=100, value=10)

# Şehir seçimi
sehirler = list(model_paths.keys())  # Model dosyalarıyla eşleşen şehirler
sehir = st.sidebar.selectbox("Şehir", sehirler)

# Seçilen şehre göre model yükleme
if sehir:
    model = joblib.load(model_paths[sehir])

# Kullanıcıdan girdileri alıp modele uygun formata dönüştürme
if st.sidebar.button("Tahmin Yap"):
    input_data = np.array([[metrekare, oda_sayisi, bina_yasi]])

    # Modeli kullanarak tahmin yapma
    tahmin = model.predict(input_data)

    # Sonucu ekranda gösterme
    st.success(f"Tahmini Ev Fiyatı: {tahmin[0]:.2f} TL")





























