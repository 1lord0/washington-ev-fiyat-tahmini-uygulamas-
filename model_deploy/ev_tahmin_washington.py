


import pandas as pd

import numpy as np
df=pd.read_csv(r"C:\Users\eren\Desktop\makine sadi evren sc\archive_3\data.csv")

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

best_params_city={'Shoreline': {'alpha': 0.7742636826811278, 'l1_ratio': 0.8888888888888888},
 'Seattle': {'alpha': 0.0001, 'l1_ratio': 1.0},
 'Bellevue': {'alpha': 0.004641588833612782, 'l1_ratio': 0.5555555555555556},
 'Redmond': {'alpha': 0.0001, 'l1_ratio': 1.0},
 'Sammamish': {'alpha': 0.7742636826811278, 'l1_ratio': 0.2222222222222222},
 'Auburn': {'alpha': 0.7742636826811278, 'l1_ratio': 0.3333333333333333},
 'Federal Way': {'alpha': 0.05994842503189409, 'l1_ratio': 0.3333333333333333},
 'Kirkland': {'alpha': 0.0001, 'l1_ratio': 1.0},
 'Issaquah': {'alpha': 0.0001, 'l1_ratio': 1.0},
 'Woodinville': {'alpha': 10.0, 'l1_ratio': 0.0},
 'Renton': {'alpha': 0.7742636826811278, 'l1_ratio': 0.1111111111111111},
 'Sammamish': {'alpha': 0.7742636826811278, 'l1_ratio': 0.2222222222222222},
 'Seattle':{'alpha': 0.0001, 'l1_ratio': 1.0},
 'Shoreline':{'alpha': 0.7742636826811278, 'l1_ratio': 0.8888888888888888},
  'Woodinville':{'alpha': 10.0, 'l1_ratio': 0.0}}
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


predictions_by_city = {}



for city in df['city'].unique():
    df_city = df[df['city'] == city]
    n_samples = len(df_city)

    if city not in best_params_by_city or n_samples < 2:
        print(f"Yetersiz veri nedeniyle {city} atlandı.")
        continue  

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
import numpy as np
import joblib  # Modelinizi yüklemek için
model1=model_deploy/woodinville_model.pkl
# Kullanıcıya hangi modeli kullanacağı sorulacak
model_secimi = st.sidebar.selectbox("Kullanmak İstediğiniz Modeli Seçin", ["Model 1", "Model 2", "Model 3"])

# Modeli yükleme (seçilen modele göre)
if model_secimi == "Model 1":
    model = joblib.load('model_1.pkl')
elif model_secimi == "Model 2":
    model = joblib.load('model_2.pkl')
elif model_secimi == "Model 3":
    model = joblib.load('model_3.pkl')

# Şehir listesi
sehirler = ['Washington', 'New York', 'Los Angeles', 'Chicago', 'Miami']

# Kullanıcıdan girdileri alma
st.sidebar.header("Ev Özelliklerini Girin")

# Örnek girdiler
metrekare = st.sidebar.number_input("Metrekare (m²)", min_value=50, max_value=500, value=100)
oda_sayisi = st.sidebar.number_input("Oda Sayısı", min_value=1, max_value=10, value=3)
bina_yasi = st.sidebar.number_input("Bina Yaşı", min_value=0, max_value=100, value=10)

# Şehir seçimi
sehir = st.sidebar.selectbox("Şehir", sehirler)

# Şehri sayısal değere dönüştürme
sehir_mapping = {sehir: idx for idx, sehir in enumerate(sehirler)}
sehir_encoded = sehir_mapping[sehir]

# Tahmin yapma butonu
if st.sidebar.button("Tahmin Yap"):
    # Kullanıcı girdilerini modele uygun formata dönüştürme
    input_data = np.array([[metrekare, oda_sayisi, bina_yasi, sehir_encoded]])
    
    # Tahmin yapma
    tahmin = model.predict(input_data)
    
    # Sonucu ekrana yazdırma
    st.success(f"Tahmini Ev Fiyatı: {tahmin[0]:.2f} TL")




























"""





# Streamlit uygulamasının başlığı
st.title("🏠 Ev Fiyat Tahmin Uygulaması")
st.write("Bu uygulama, Elastic Net modeli kullanarak ev fiyatlarını tahmin eder.")

# Modeli yükleme
model = joblib.load('C:\\Users\\eren\\Desktop\\model_deploy\\ev_fiyat_modeli.pkl    ')

# Şehir listesi (veri setinizdeki şehirler)
sehirler = [
    "Seattle", "Renton", "Bellevue", "Redmond", "Issaquah", "Kirkland", "Kent", 
    "Auburn", "Sammamish", "Federal Way", ""Shoreline", "Woodinville"
]



"if sehir == "Seattle":
    model = joblib.load('C:\\Users\\eren\\Desktop\\model_deploy\\seattle_model.pkl')
elif sehir == "Renton":
    model = joblib.load('C:\\Users\\eren\\Desktop\\model_deploy\\renton_model.pkl')
elif sehir == "Bellevue":
    model = joblib.load('C:\\Users\\eren\\Desktop\\model_deploy\\bellevue_model.pkl')


# Kullanıcıdan girdileri alma
st.sidebar.header("Ev Özelliklerini Girin")

# Örnek girdiler
metrekare = st.sidebar.number_input("Metrekare (m²)", min_value=50, max_value=500, value=100)
oda_sayisi = st.sidebar.number_input("Oda Sayısı", min_value=1, max_value=10, value=3)
bina_yasi = st.sidebar.number_input("Bina Yaşı", min_value=0, max_value=100, value=10)

# Şehir seçimi
sehir = st.sidebar.selectbox("Şehir", sehirler)

# Şehri sayısal değere dönüştürme
sehir_mapping = {sehir: idx for idx, sehir in enumerate(sehirler)}
sehir_encoded = sehir_mapping[sehir]

# Tahmin yapma butonu
if st.sidebar.button("Tahmin Yap"):
    # Kullanıcı girdilerini modele uygun formata dönüştürme
    input_data = np.array([[metrekare, oda_sayisi, bina_yasi, sehir_encoded]])
    
    # Tahmin yapma
    tahmin = model.predict(input_data)
    
    # Sonucu ekrana yazdırma
    st.success(f"Tahmini Ev Fiyatı: {tahmin[0]:.2f} TL")""
"""





