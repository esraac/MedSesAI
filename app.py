from flask import Flask, request, jsonify
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import os
import json
import joblib
import uuid 

app = Flask(__name__)
print("VoiceEncoder yükleniyor...")
encoder = VoiceEncoder()
print("VoiceEncoder hazır.")

@app.route('/api/voice/encode', methods=['POST'])
def encode_voice():
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası yok'}), 400
    
    file = request.files['file']
    # UZANTI DÜZELTİLDİ: .wav yapıldı
    filename = f"temp_encode_{uuid.uuid4()}.wav" 
    
    try:
        file.save(filename)
        wav = preprocess_wav(filename)
        embedding = encoder.embed_utterance(wav)
        return jsonify({'voice_vector': embedding.tolist()})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)

@app.route('/api/voice/verify', methods=['POST'])
def verify_voice():
    if 'file' not in request.files:
        return jsonify({'error': 'Ses dosyası eksik'}), 400
        
    file = request.files['file']
    saved_vector_str = request.form.get('saved_vector')
    
    if not saved_vector_str:
        return jsonify({'error': 'Kayıtlı vektör gelmedi'}), 400

    filename = f"temp_verify_{uuid.uuid4()}.wav"

    try:
        saved_embedding = np.array(json.loads(saved_vector_str))
        
        file.save(filename)
        wav = preprocess_wav(filename)
        new_embedding = encoder.embed_utterance(wav)
        
        similarity = np.inner(new_embedding, saved_embedding)
        is_match = bool(similarity > 0.75) # Daha güvenli boolean dönüşümü
        
        return jsonify({
            'match': is_match,
            'confidence': float(similarity) # ANAHTAR İSMİ DEĞİŞTİ (Java logları için)
        })
        
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
