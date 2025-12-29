import os
import re
import sys
import math
from collections import Counter
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# =============================================================================
# 1. AYARLAR (DEEPSEEK - BACKEND MODE)
# =============================================================================
load_dotenv()

# DeepSeek Anahtarını Buraya (veya .env dosyasına) koy
DEEPSEEK_API_KEY = "sk-4109a78af41142d2b7a0d89427d0ca97" # Senin Key'in

if not DEEPSEEK_API_KEY:
    print("\n❌ HATA: DeepSeek API Key bulunamadı.")
    sys.exit()

# Client Tanımlaması
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com" # ⚠️ KRİTİK AYAR
)

print("\n🚀 AI Eczacı Asistanı (Terminal/CLI Modu) Başlatılıyor...")
print("🔧 Motor: DeepSeek-V3 | Mod: Text Only\n")

# =============================================================================
# 2. RAG MOTORU (SAF PYTHON)
# =============================================================================
class SimpleRAG:
    def __init__(self):
        self.vector_store = []
        self.loaded_files = [] 

    def get_embedding(self, text):
        words = re.findall(r'\w+', text.lower())
        return Counter(words)

    def _cosine_similarity(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        return float(numerator) / denominator if denominator else 0.0

    def ingest(self, text, source_name=""):
        if source_name not in self.loaded_files:
            self.loaded_files.append(source_name)

        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 10]

        WINDOW_SIZE = 10  
        STEP = 3          
        
        for i in range(0, len(sentences), STEP):
            window = sentences[i : i + WINDOW_SIZE]
            chunk_text = " ".join(window)
            
            self.vector_store.append({
                "source": source_name,
                "content": f"[{source_name}] {chunk_text}",
                "vector": self.get_embedding(chunk_text)
            })

    def retrieve(self, query, top_k=5): 
        query_vec = self.get_embedding(query)
        candidates = self.vector_store
        scores = []
        for item in candidates:
            score = self._cosine_similarity(query_vec, item["vector"])
            scores.append((score, item["content"]))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        
        if not scores or scores[0][0] < 0.1:
            return "VERİTABANINDA YOK"

        return "\n\n".join([item[1] for item in scores[:top_k]])

# --- PDF YÜKLEME ---
rag_engine = SimpleRAG()
PDF_KLASORU = "./pdf_data"

def pdf_yukle():
    if not os.path.exists(PDF_KLASORU):
        print(f"⚠️ UYARI: '{PDF_KLASORU}' klasörü yok. Sadece genel bilgi ile çalışacak.")
        return

    print("📂 PDF Dosyaları Taranıyor...")
    dosyalar = [f for f in os.listdir(PDF_KLASORU) if f.endswith('.pdf')]
    
    if not dosyalar:
        print("ℹ️ Klasör boş.")
        return

    for dosya in dosyalar:
        try:
            reader = PdfReader(os.path.join(PDF_KLASORU, dosya))
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            rag_engine.ingest(text, source_name=dosya.replace(".pdf", "").upper())
            print(f"   ✅ Yüklendi: {dosya}")
        except Exception as e:
            print(f"   ❌ Hata ({dosya}): {e}")
            
    print("💾 Veritabanı Hazır!\n")

pdf_yukle()

# =============================================================================
# 3. AJAN FONKSİYONU
# =============================================================================
def ask_agent(question):
    # 1. Önce RAG Araması Yap
    print(f"🔎 Araştırılıyor: '{question}'...")
    context = rag_engine.retrieve(question)
    
    system_prompt = """
    Sen Uzman bir Eczacı Asistanısın (Strict Mode).
    GÖREVLERİN:
    1. Sana verilen 'DOKÜMANLAR' kısmındaki bilgiyi kullan.
    2. Eğer ilaç veritabanında YOKSA "Veritabanında bulunamadı" de. Uydurma.
    3. Kullanıcıyı 'HAYIR, SAKIN, RİSKLİ' gibi kelimelerle net uyar.
    """
    
    user_message = f"SORU: {question}\n\nDOKÜMANLAR:\n{context}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # DeepSeek Modeli
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0, # Tutarlılık için
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Hatası: {e}"

# =============================================================================
# 4. ÇALIŞTIRMA DÖNGÜSÜ (CLI)
# =============================================================================
if __name__ == "__main__":
    print("💡 Çıkmak için 'q' veya 'exit' yazın.\n")
    
    while True:
        soru = input(f"\n💊 Sorunuz: ")
        
        if soru.lower() in ["q", "exit", "cikis"]:
            print("👋 Güle güle!")
            break
        
        if not soru: continue
        
        cevap = ask_agent(soru)
        print(f"\n🤖 Asistan: {cevap}")
        print("-" * 50)