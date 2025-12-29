import google.generativeai as genai
import os
from dotenv import load_dotenv

# .env dosyasını yüklemeyi dene
load_dotenv()

# Eğer .env'den gelmezse, buraya manuel olarak tırnak içine yaz:
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
# Örn: GOOGLE_API_KEY = "AIzaSy....."

if not GOOGLE_API_KEY:
    print("❌ HATA: API Key bulunamadı! Kodu düzenleyip key'i yapıştırın.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

    print("\n🔍 Hesabınızda Kullanılabilir Modeller Aranıyor...\n")
    try:
        found_vision = False
        for m in genai.list_models():
            # Sadece içerik üretebilen modelleri listele
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ {m.name}")
                if "vision" in m.name or "flash" in m.name:
                    found_vision = True
        
        if not found_vision:
            print("\n⚠️ UYARI: Listenizde hiç 'Vision' veya 'Flash' modeli görünmüyor.")
            print("Çözüm: https://aistudio.google.com adresinden yeni bir API Key alın.")
            
    except Exception as e:
        print(f"\n❌ Bağlantı Hatası: {e}")
        print("Çözüm: 'pip install --upgrade google-generativeai' komutunu çalıştırın.")