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
print("VoiceEncoder hazır.")

print("SpeechRecognition ayarlanıyor...")
recognizer = sr.Recognizer()
print("SpeechRecognition hazır.")

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

print("NLP Modeli yükleniyor...")
try:
    model = joblib.load('model.pkl')
    print("NLP Modeli yüklendi.")
except Exception as e:
    print(f"UYARI: model.pkl bulunamadı veya yüklenemedi! Hata: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model sunucuda yüklü değil'}), 500

    try:
        data = request.get_json(force=True) 
        user_text = data.get('text', '')

        print(f"GELEN ŞİKAYET: {user_text}")

        if not user_text:
            return jsonify({'error': 'Metin boş geldi'}), 400

        user_text_clean = user_text.lower()
        probs = model.predict_proba([user_text_clean])[0]
        max_prob = probs.max() 
        prediction = model.predict([user_text_clean])[0] 

        print(f"Tahmin: {prediction}, Güven: {max_prob:.2f}")

        return jsonify({
            'semptom': user_text,
            'onerilen_klinik': prediction,
            'guven_orani': float(max_prob)
        })

    except Exception as e:
        print(f"SUNUCU HATASI: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
