import pandas as pd
import json
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

print('--- BERT (DISTILBERT) EĞİTİM MODÜLÜ (GÜNCELLENMİŞ) ---')

# 1. Veri Yükleme ve Hazırlık
def read_csv_safe(file, sep=','):
    try:
        return pd.read_csv(file, sep=sep, encoding='utf-8-sig', on_bad_lines='skip')
    except:
        try:
            return pd.read_csv(file, sep=sep, encoding='utf-8', on_bad_lines='skip')
        except:
            return pd.read_csv(file, sep=sep, encoding='cp1254', on_bad_lines='skip')

print('Veriler yükleniyor...')
# Kullanıcı tarafından çevrilen veya genişletilen en güncel veri setini seç
hastalik_file = 'hastalık.csv' if os.path.exists('hastalık.csv') else 'hastalık.csv'
print(f'Kaynak dosya: {hastalik_file}')

df_hastalik_raw = read_csv_safe(hastalik_file, sep=';')
df_poliklinik_raw = read_csv_safe('poliklinik_tr.csv', sep=',')

# Poliklinik Map (Hastalık -> Bölüm listesi)
hastalik_bolum_map = {}
bolum_sutunu = df_poliklinik_raw.columns[0]

# Genişletilmiş düzeltme haritası
disease_fixes = {
    "astım": "bronşiyal astım",
    "mide ülseri": "peptik ülser hastalığı",
    "yüksek tansiyon": "hipertansiyon",
    "zatüre": "akciğer iltihaplanması",
    "zatürre": "akciğer iltihaplanması",
    "basur": "hemoroid basur",
    "panik atak": "panik bozukluk",
    "akut sinüzit": "sinüzit",
    "sivilce": "akne",
    "ülser": "peptik ülser hastalığı",
    "reflü": "gerd",
    "kalp krizi": "miyokard enfarktüsü",
    "felç": "cerebrovascular accident",
    "beyin kanaması": "felç (beyin kanaması)",
    "kireçlenme": "osteoartrit",
    "ek iltihabı": "artrit"
}

for index, row in df_poliklinik_raw.iterrows():
    bolum = str(row[bolum_sutunu]).strip()
    for col in df_poliklinik_raw.columns:
        if col.lower().startswith('hastalık'):
            h = row[col]
            if pd.notna(h) and str(h).strip() != '':
                h_str = str(h).strip().lower()
                if h_str not in hastalik_bolum_map: hastalik_bolum_map[h_str] = []
                if bolum not in hastalik_bolum_map[h_str]: hastalik_bolum_map[h_str].append(bolum)
                
                # Ters eşleşme için fixes kullan
                for orig, fixed in disease_fixes.items():
                    if fixed == h_str:
                        if orig not in hastalik_bolum_map: hastalik_bolum_map[orig] = []
                        if bolum not in hastalik_bolum_map[orig]: hastalik_bolum_map[orig].append(bolum)

# JSON olarak kaydet
with open('hastalık_bolum_map.json', 'w', encoding='utf-8') as f:
    json.dump(hastalik_bolum_map, f, ensure_ascii=False, indent=2)

# BERT için metin temizleme
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# Veri setlerini oluştur
rows_hastalik = []
rows_poliklinik = []

# Sütun isimlerini bul (Büyük/Küçük harf duyarlı olabilir)
has_col = 'Hastalık' if 'Hastalık' in df_hastalik_raw.columns else df_hastalik_raw.columns[0]
semp_col = 'Semptomlar' if 'Semptomlar' in df_hastalik_raw.columns else df_hastalik_raw.columns[1]

for index, row in df_hastalik_raw.iterrows():
    hastalik = str(row[has_col]).strip().lower()
    semptomlar_str = str(row[semp_col]).strip()
    text = clean_text(semptomlar_str)
    
    if text:
        rows_hastalik.append({'text': text, 'label': hastalik})
        bolumler = hastalik_bolum_map.get(hastalik, [])
        # Eğer tam eşleşme yoksa fixes içinde ara
        if not bolumler:
            fixed_name = disease_fixes.get(hastalik)
            if fixed_name:
                bolumler = hastalik_bolum_map.get(fixed_name, [])
        
        for bolum in bolumler:
            rows_poliklinik.append({'text': text, 'label': bolum})

df_has = pd.DataFrame(rows_hastalik).drop_duplicates()
df_pol = pd.DataFrame(rows_poliklinik).drop_duplicates()

print(f'Hastalık Verisi: {len(df_has)}, Poliklinik Verisi: {len(df_pol)}')

# BERT Eğitim Fonksiyonu
def train_bert_model(df, output_dir, model_name="dbmdz/distilbert-base-turkish-cased"):
    if len(df) < 10:
        print(f"Uyarı: {output_dir} için yetersiz veri ({len(df)} satır). Eğitim atlanıyor.")
        return

    print(f'\n--- {output_dir} Eğitimi Başlıyor ({len(df)} örnek) ---')
    
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    num_labels = len(le.classes_)
    
    with open(f'{output_dir}_classes.json', 'w', encoding='utf-8') as f:
        json.dump(le.classes_.tolist(), f, ensure_ascii=False)
    
    dataset = Dataset.from_pandas(df[['text', 'label_id']])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")
    tokenized_dataset.set_format("torch")
    
    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in enumerate(le.classes_)}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Eğitim cihazı: {device}")
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=f"./results_{output_dir}",
        num_train_epochs=5, # Veri çok arttığı için epoch sayısını düşürdük
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        push_to_hub=False,
        report_to="none",
        save_strategy="no" # Disk alanından tasarruf
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model {output_dir} klasörüne kaydedildi.")

# Eğitimleri çalıştır
train_bert_model(df_has, "bert_hastalik")
train_bert_model(df_pol, "bert_poliklinik")

print('\nBAŞARILI: BERT modelleri eğitildi!')
