import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

print("--- MEDSES AI EĞİTİM MODÜLÜ ---")

# 1. Veriyi Oku
try:
    data = pd.read_csv('veri.csv')
    print("Veri başarıyla okundu.")
except FileNotFoundError:
    print("HATA: 'veri.csv' bulunamadı!")
    exit()

# 2. Veri Temizliği ve Analizi
data.dropna(inplace=True) # Boş satırları sil
data['text'] = data['text'].str.lower() # Hepsini küçük harfe çevir

print(f"\nToplam Veri Sayısı: {len(data)}")
print("Kliniklere Göre Veri Dağılımı (Dengesizse düzeltin!):")
print(data['label'].value_counts())
print("-" * 30)

# 3. Modeli Hazırla
# TfidfVectorizer: Metni sayısal vektörlere çevirir
# MultinomialNB: Sınıflandırma yapar
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 4. Modeli Eğit
print("Model eğitiliyor...")
model.fit(data['text'], data['label'])

# 5. Test Et
ornek_cumleler = ["gözlerim ağrıyor", "karnım çok kötü", "başım dönüyor"]
print("\n--- HIZLI TEST SONUÇLARI ---")
for cumle in ornek_cumleler:
    tahmin = model.predict([cumle])[0]
    olasilik = model.predict_proba([cumle]).max()
    print(f"Cümle: '{cumle}' -> Tahmin: {tahmin} (Güven: %{olasilik*100:.1f})")

# 6. Kaydet
joblib.dump(model, 'model.pkl')
print("\nBAŞARILI: Model 'model.pkl' olarak kaydedildi!")