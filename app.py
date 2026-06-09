from flask import Flask, request, jsonify
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
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
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    # GPU varsa kullan (device=0), yoksa CPU (device=-1)
    device = 0 if torch.cuda.is_available() else -1
    
    def load_custom_pipeline(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if "token_type_ids" in tokenizer.model_input_names:
            tokenizer.model_input_names.remove("token_type_ids")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    
    pipe_hastalik = load_custom_pipeline("bert_hastalik")
    pipe_poliklinik = load_custom_pipeline("bert_poliklinik")
    
    try:
        pipe_aciliyet = load_custom_pipeline("bert_aciliyet")
        print(f"✓ Aciliyet Modeli yüklendi (Cihaz: {'GPU' if device == 0 else 'CPU'})")
    except Exception as e:
        print(f"⚠ Aciliyet Modeli yüklenemedi, sistem normal çalışmaya devam edecek: {e}")
        pipe_aciliyet = None
    
    print(f"✓ BERT Modelleri yüklendi (Cihaz: {'GPU' if device == 0 else 'CPU'})")
except Exception as e:
    print(f"❌ BERT Modelleri yüklenemedi: {e}")
    pipe_hastalik = None
    pipe_poliklinik = None
    pipe_aciliyet = None

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
        
        similarity = 1 - cosine(new_embedding, saved_embedding)
        is_speaker_match = bool(similarity > 0.70) 
        
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


@app.route('/predict', methods=['POST'])
def predict():
    if pipe_hastalik is None or pipe_poliklinik is None:
        return jsonify({'error': 'BERT modelleri sunucuda yüklü değil'}), 500

    try:
        data = request.get_json(force=True) 
        user_text = data.get('text', '')

        if not user_text:
            return jsonify({'error': 'Metin boş geldi'}), 400

        # BERT için temel temizlik 
        user_text_clean = user_text.lower().strip()
        print(f"BERT GİRDİSİ: {user_text_clean}")
        
        # ilk 5 al 
        res_has_top3 = pipe_hastalik([user_text_clean], top_k=5)[0]
        
        prob_dict = {}
        if pipe_poliklinik is not None:
            res_pol_all = pipe_poliklinik([user_text_clean], top_k=None)[0]
            prob_dict = {item['label']: item['score'] for item in res_pol_all}
            
        print(f"\n--- ENSEMBLE RE-RANKING (ÇAPRAZ DOĞRULAMA) ---")
        
        best_score = -1.0
        final_hastalik = res_has_top3[0]['label']
        final_klinik = None
        final_hastalik_prob = res_has_top3[0]['score']
        
        for idx, has_item in enumerate(res_has_top3):
            hastalik_adi = has_item['label']
            h_score = has_item['score']
            
            ilgili_klinikler = hastalık_bolum_map.get(hastalik_adi, [])
            
            max_p_score = 0
            en_iyi_klinik = None
            
            for pk in ilgili_klinikler:
                p_score = prob_dict.get(pk, 0.0)
                if p_score > max_p_score:
                    max_p_score = p_score
                    en_iyi_klinik = pk
            
            # çapraz doğrulama
            # eğer hastalıkla eşleşen hiçbir poliklinik yoksa 0.01 ceza verilir
            cross_score = h_score * max_p_score if max_p_score > 0 else h_score * 0.01 
            
            print(f"[{idx+1}] {hastalik_adi} ({h_score:.3f}) -> {en_iyi_klinik} ({max_p_score:.3f}) | Çapraz Skor: {cross_score:.4f}")
            
            if cross_score > best_score:
                best_score = cross_score
                final_hastalik = hastalik_adi
                final_klinik = en_iyi_klinik
                final_hastalik_prob = h_score
                
        if not final_klinik:
            final_klinik = "İç Hastalıkları (Dahiliye)"
            print("⚠ Klinik bulunamadı, İç Hastalıkları (Dahiliye) önerildi.")
            
        print(f"NİHAİ KARAR -> Hastalık: {final_hastalik}, Klinik: {final_klinik}")
        print("----------------------------------------------\n")
        
    
        clinic_scores = {}
        if pipe_poliklinik is not None:
            for pk, p_score in prob_dict.items():
                clinic_scores[pk] = 0.01 * p_score # Başlangıçta cezalı skor
                
        for has_item in res_has_top3:
            hastalik_adi = has_item['label']
            h_score = has_item['score']
            ilgili_klinikler = hastalık_bolum_map.get(hastalik_adi, []) if hastalık_bolum_map else []
            for pk in ilgili_klinikler:
                p_score = prob_dict.get(pk, 0.0)
                cross_score = h_score * p_score
                if cross_score > clinic_scores.get(pk, 0.0):
                    clinic_scores[pk] = cross_score
                    
        sorted_clinics = sorted(clinic_scores.items(), key=lambda x: x[1], reverse=True)
        top_3_clinics = [item[0] for item in sorted_clinics[:3]]
        
        default_fallback = ["İç Hastalıkları (Dahiliye)", "Aile Hekimliği", "Acil Tıp"]
        for fallback in default_fallback:
            if len(top_3_clinics) < 3 and fallback not in top_3_clinics:
                top_3_clinics.append(fallback)
                
        
        if final_klinik:
            if final_klinik in top_3_clinics:
                top_3_clinics.remove(final_klinik)
            top_3_clinics.insert(0, final_klinik)
            top_3_clinics = top_3_clinics[:3]

        # aciliyet Tahmini
        aciliyet_durumu = "NORMAL"
        acil_uyarisi = ""
        
        if pipe_aciliyet is not None:
            res_acil = pipe_aciliyet([user_text_clean], top_k=1)[0]
            aciliyet_durumu = res_acil[0]['label']
            print(f"Aciliyet Tahmini: {aciliyet_durumu} (Güven: {res_acil[0]['score']:.2f})")
            
            if aciliyet_durumu in ["ACİL", "ÇOK ACİL"]:
                acil_uyarisi = f"DİKKAT: Belirtileriniz {aciliyet_durumu} sinyali veriyor. Lütfen vakit kaybetmeden 112'yi arayın veya en yakın Acil Servise başvurun!"
                final_klinik = "Acil Tıp"
                print("⚠ Acil durum tespit edildi, poliklinik yönlendirmesi 'Acil Tıp' olarak ezildi.")

       
        if final_klinik:
            if final_klinik in top_3_clinics:
                top_3_clinics.remove(final_klinik)
            top_3_clinics.insert(0, final_klinik)
            top_3_clinics = top_3_clinics[:3]

        hastalik_tahmin = final_hastalik
        secilen_klinikler = top_3_clinics
        max_hastalik_prob = final_hastalik_prob

        return jsonify({
            'semptom': user_text,
            'hastalik_tahmin': hastalik_tahmin,
            'onerilen_klinik': secilen_klinikler,
            'guven_orani': float(max_hastalik_prob),
            'aciliyet_durumu': aciliyet_durumu,
            'acil_uyarisi': acil_uyarisi
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
