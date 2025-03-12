pip install scikit-learn
pip install pandas
pip install numpy



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


[ df.drop(i,axis=0,inplace=True) for i in df['city'].value_counts() if i<100 ]

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

best_params_by_city = {}

for city in df['city'].unique():
    df_city = df[df['city'] == city]
    n_samples = len(df_city)  # O şehirdeki veri sayısı

    if n_samples < 2:
        print(f"Yetersiz veri nedeniyle {city} atlandı.")
        continue  

    print(f"Şehir: {city}")

    X = df_city.drop(columns=['price', 'city'])  
    y = df_city['price']  

    if n_samples < 10:
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'alpha': np.logspace(-4, 1, 10),  
        'l1_ratio': np.linspace(0, 1, 10)  
    }

    # Veri sayısına göre `cv` değerini ayarla
    cv_value = min(3, n_samples)  # Örnek sayısından büyük olamaz

    model = ElasticNet()
    
    # Eğer veri sayısı 1 ise GridSearch kullanmadan modeli eğit
    if n_samples == 1:
        model.fit(X_train, y_train)
        best_params = {'alpha': model.alpha, 'l1_ratio': model.l1_ratio}
    else:
        grid_search = GridSearchCV(model, param_grid, cv=cv_value, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

    best_params_by_city[city] = best_params
    print(f"Şehir: {city}, En iyi parametreler: {best_params}")

print("\nTüm şehirler için en iyi parametreler:")
print(best_params_by_city)




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

# Tahmin sonuçlarını göster
print("\n📊 Tüm şehirler için tahmin sonuçları:")
for city, results in predictions_by_city.items():
    print(f"\n🏙 Şehir: {city}")
    print(f"📌 Gerçek: {results['Gerçek Değerler'][:5]}")
    print(f"🔮 Tahmin: {results['Tahminler'][:5]}")
    print(f"📉 MAE: {results['MAE']:.2f}")
    print(f"📊 MSE: {results['MSE']:.2f}")
    print(f"📈 R²: {results['R2']:.2f}")


import pickle






import streamlit as st
import joblib
import numpy as np
import pandas as pd






import joblib

import joblib

import joblib
import os

# Modellerin kaydedileceği klasörü belirle
save_dir = "C:\\Users\\eren\\Desktop\\model_deploy"
os.makedirs(save_dir, exist_ok=True)  # Klasör yoksa oluştur

# Her şehir için modeli kaydet
for city, model in best_params_by_city.items():
    file_path = os.path.join(save_dir, f"{city.lower()}_model.pkl")  # Dosya yolunu oluştur
    joblib.dump(model, file_path)
    print(f"{file_path} dosyasına kaydedildi!")





import streamlit as st
import numpy as np
import joblib  # Modelinizi yüklemek için

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





