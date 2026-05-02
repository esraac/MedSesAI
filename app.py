from flask import Flask, request, jsonify
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import os
import json
import joblib
import uuid 
import speech_recognition as sr

app = Flask(__name__)

print("VoiceEncoder yükleniyor...")
encoder = VoiceEncoder()
print("✓ VoiceEncoder hazır.")

print("SpeechRecognition ayarlanıyor...")
recognizer = sr.Recognizer()
print("✓ SpeechRecognition hazır.")

print("BERT Modelleri yükleniyor...")
try:
    from transformers import pipeline
    import torch
    
    # GPU varsa kullan (device=0), yoksa CPU (device=-1)
    device = 0 if torch.cuda.is_available() else -1
    
    pipe_hastalik = pipeline("text-classification", model="bert_hastalik", device=device)
    pipe_poliklinik = pipeline("text-classification", model="bert_poliklinik", device=device)
    
    print(f"✓ BERT Modelleri yüklendi (Cihaz: {'GPU' if device == 0 else 'CPU'})")
except Exception as e:
    print(f"❌ BERT Modelleri yüklenemedi: {e}")
    pipe_hastalik = None
    pipe_poliklinik = None

try:
    with open('hastalık_bolum_map.json', 'r', encoding='utf-8') as f:
        hastalık_bolum_map = json.load(f)
    print(f"✓ Hastalık-Poliklinik Mapping yüklendi ({len(hastalık_bolum_map)} hastalık)")
except Exception as e:
    print(f"❌ hastalık_bolum_map.json yüklenemedi: {e}")
    hastalık_bolum_map = None

@app.route('/api/voice/encode', methods=['POST'])
def encode_voice():
    files = request.files.getlist('files')
    
    if not files or len(files) == 0:
        if 'file' in request.files:
            files = [request.files['file']]
        else:
            return jsonify({'error': 'Ses dosyası yok'}), 400
    
    embeddings = []
    temp_files = []
    
    try:
        for file in files:
            filename = f"temp_encode_{uuid.uuid4()}.wav"
            temp_files.append(filename)
            file.save(filename)
            
            wav = preprocess_wav(filename)
            embedding = encoder.embed_utterance(wav)
            embeddings.append(embedding)
            
        if len(embeddings) > 1:
            final_embedding = np.mean(embeddings, axis=0)
        else:
            final_embedding = embeddings[0]
            
        return jsonify({'voice_vector': final_embedding.tolist()})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        for filename in temp_files:
            if os.path.exists(filename):
                os.remove(filename)

@app.route('/api/voice/verify', methods=['POST'])
def verify_voice():
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası eksik'}), 400
        
    file = request.files['file']
    saved_vector_str = request.form.get('saved_vector')
    challenge_code = request.form.get('challenge_code') 
    if not saved_vector_str:
        return jsonify({'error': 'Kayıtlı vektör gelmedi'}), 400

    filename = f"temp_verify_{uuid.uuid4()}.wav"

    try:
        saved_embedding = np.array(json.loads(saved_vector_str))
        
        file.save(filename)
        
        wav = preprocess_wav(filename)
        new_embedding = encoder.embed_utterance(wav)
        
        similarity = np.inner(new_embedding, saved_embedding)
        is_speaker_match = bool(similarity > 0.60) 
        
        print(f"\n--- SES DOĞRULAMA BAŞLADI ---")
        print(f"Similarity (Benzerlik Skoru): {similarity:.4f} (Eşik: 0.70)")
        print(f"Speaker Match: {is_speaker_match}")

        stt_match = False
        recognized_text = ""
        
        if challenge_code:
            try:
                with sr.AudioFile(filename) as source:
                    audio_data = recognizer.record(source)
                    recognized_text = recognizer.recognize_google(audio_data, language="tr-TR")
                   
                    text_lower = recognized_text.lower()
                    number_map = {
                        "sıfır": "0", "bir": "1", "iki": "2", "üç": "3", "üc": "3",
                        "dört": "4", "dort": "4", "beş": "5", "bes": "5", 
                        "altı": "6", "alti": "6", "yedi": "7", "sekiz": "8", "dokuz": "9"
                    }
                    extracted_digits = ""
                    for char in recognized_text:
                        if char.isdigit():
                            extracted_digits += char
                            
                    if not extracted_digits:
                        words = text_lower.replace(",", " ").replace(".", " ").split()
                        for word in words:
                            if word in number_map:
                                extracted_digits += number_map[word]
                   
                    if challenge_code in extracted_digits or challenge_code in text_lower.replace(" ", ""):
                        stt_match = True
                    if extracted_digits == challenge_code:
                        stt_match = True
                    
                    print(f"Beklenen Sayı: {challenge_code}")
                    print(f"Algılanan Metin: {recognized_text}")
                    print(f"Ayıklanan Rakamlar: {extracted_digits}")
                    print(f"STT Match: {stt_match}")
                    print(f"--- DOĞRULAMA BİTTİ ---\n")
                        
            except sr.UnknownValueError:
                recognized_text = "Sesi metne çeviremedi (anlaşılamadı)"
            except sr.RequestError as e:
                recognized_text = f"STT Servis Hatası: {e}"
        else:
            stt_match = True 

        return jsonify({
            'match': is_speaker_match,         
            'stt_match': stt_match,            
            'recognized_text': recognized_text, 
            'confidence': float(similarity) 
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)

@app.route('/api/voice/stt', methods=['POST'])
def recognize_stt():
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası eksik'}), 400
        
    file = request.files['file']
    filename = f"temp_stt_{uuid.uuid4()}.wav"
    
    try:
        file.save(filename)
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="tr-TR")
            
            # Sayıları ayıkla (TC için)
            digits = "".join([c for c in text if c.isdigit()])
            
            print(f"\n--- STT TANIMA ---")
            print(f"Algılanan Metin: {text}")
            print(f"Ayıklanan Rakamlar (TC): {digits}")
            print(f"------------------\n")

            return jsonify({
                'text': text,
                'digits': digits
            })
    except sr.UnknownValueError:
        return jsonify({'error': 'Ses anlaşılamadı'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filename):
            os.remove(filename)

# ============ NLP TAHMİN ENDPOINTS ============

@app.route('/predict', methods=['POST'])
def predict():
    if pipe_hastalik is None or pipe_poliklinik is None:
        return jsonify({'error': 'BERT modelleri sunucuda yüklü değil'}), 500

    try:
        data = request.get_json(force=True) 
        user_text = data.get('text', '')

        print(f"\n--- YENİ AKILLI TAHMİN (2 AŞAMALI) ---")
        print(f"GELEN ŞİKAYET: {user_text}")

        if not user_text:
            return jsonify({'error': 'Metin boş geldi'}), 400

        # BERT için temel temizlik (Küçük harf ve boşluklar)
        user_text_clean = user_text.lower().strip()
        print(f"BERT GİRDİSİ: {user_text_clean}")
        
        # AŞAMA 1: BERT Hastalık Tahmini
        res_has = pipe_hastalik([user_text_clean], top_k=1)[0]
        hastalik_tahmin = res_has[0]['label']
        max_hastalik_prob = res_has[0]['score']
        print(f"Tahmin Edilen Hastalık: {hastalik_tahmin} (Güven: {max_hastalik_prob:.2f})")

        # AŞAMA 2: Poliklinik Listesi Çekimi
        poliklinikler = hastalık_bolum_map.get(hastalik_tahmin, [])
        
        secilen_klinik = None
        
        if not poliklinikler:
            secilen_klinik = hastalik_tahmin # Fallback olarak hastalık adını dön
        elif len(poliklinikler) == 1:
            secilen_klinik = poliklinikler[0]
            print(f"Tek Poliklinik Var, Doğrudan Seçildi: {secilen_klinik}")
        else:
            # AŞAMA 3: Akıllı Tie-Breaker (İhtimal Karşılaştırma)
            print(f"Birden Fazla Poliklinik Var: {poliklinikler}")
            
            # Poliklinik modelinden tüm ihtimalleri al
            res_pol_all = pipe_poliklinik([user_text_clean], top_k=None)[0]
            
            # İhtimalleri sözlüğe çevir (Hızlı arama için)
            prob_dict = {item['label']: item['score'] for item in res_pol_all}
            
            en_yuksek_ihtimal = -1
            
            for pol in poliklinikler:
                prob = prob_dict.get(pol, 0)
                print(f"  -> {pol} İhtimali: {prob:.4f}")
                if prob > en_yuksek_ihtimal:
                    en_yuksek_ihtimal = prob
                    secilen_klinik = pol
            
            if not secilen_klinik:
                secilen_klinik = poliklinikler[0]
            
            print(f"Akıllı Seçim Sonucu: {secilen_klinik} (Güven: {en_yuksek_ihtimal:.4f})")

        return jsonify({
            'semptom': user_text,
            'hastalik_tahmin': hastalik_tahmin,
            'onerilen_klinik': secilen_klinik,
            'guven_orani': float(max_hastalik_prob)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_full', methods=['POST'])
def predict_full():
    """
    FULL SİSTEM (2 AŞAMA): Semptom → Hastalık → Poliklinik
    Girdi: {"text": "başım çok ağrıyor"}
    Çıktı: {
        "hastalık": "Migren", 
        "güven": 0.95,
        "poliklinikler": ["Nöroloji", "Beyin ve Sinir Cerrahisi"]
    }
    """
    if model_hastalik is None or hastalık_bolum_map is None:
        return jsonify({'error': 'Modeller yüklü değil'}), 500

    try:
        data = request.get_json(force=True) 
        user_text = data.get('text', '')

        print(f"\n--- FULL SİSTEM (2 AŞAMA) ---")
        print(f"GELEN SEMPTOMlar: {user_text}")

        if not user_text:
            return jsonify({'error': 'Metin boş geldi'}), 400

        # AŞAMA 1: Semptom → Hastalık
        user_text_clean = user_text.lower()
        probs = model_hastalik.predict_proba([user_text_clean])[0]
        max_prob = probs.max() 
        hastalık_tahmin = model_hastalik.predict([user_text_clean])[0]

        print(f"✓ AŞAMA 1 (Semptom → Hastalık): {hastalık_tahmin} (Güven: {max_prob:.2%})")

        # AŞAMA 2: Hastalık → Poliklinik
        poliklinikler = hastalık_bolum_map.get(hastalık_tahmin, [])
        
        if not poliklinikler:
            poliklinikler = []
            print(f"⚠ Hastalık poliklinik mapping'de bulunamadı!")
        else:
            print(f"✓ AŞAMA 2 (Hastalık → Poliklinik): {', '.join(poliklinikler[:3])} ... (Toplam: {len(poliklinikler)})")

        return jsonify({
            'semptomlar': user_text,
            'hastalık': hastalık_tahmin,
            'güven_orani': float(max_prob),
            'poliklinikler': poliklinikler,
            'onerilen_poliklinik': poliklinikler[0] if poliklinikler else None
        })

    except Exception as e:
        print(f"SUNUCU HATASI: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Sağlık kontrolü"""
    return jsonify({
        'status': 'OK',
        'semptom_model': 'loaded' if model_hastalik else 'not_loaded',
        'poliklinik_model': 'loaded' if model_poliklinik else 'not_loaded',
        'mapping': 'loaded' if hastalık_bolum_map else 'not_loaded'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
