import streamlit as st
import os
import re
import base64
from openai import OpenAI
from pypdf import PdfReader
from collections import Counter
import math
import io
import requests
import json
import urllib3
import time
from dotenv import load_dotenv
import google.generativeai as genai

# =============================================================================
# 1. AYARLAR & API KEYLER
# =============================================================================
st.set_page_config(page_title="AI Eczacı Pro v3.6", page_icon="💊", layout="wide")
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- ANAHTARLAR ---
DEEPSEEK_API_KEY = "sk-4109a78af41142d2b7a0d89427d0ca97" 
GOOGLE_API_KEY = "AIzaSyDCDd5-p94w0cYiLrEjuZ3YJmEGuX6aJOc" 

# 1. DeepSeek Ayarı (BEYİN)
if not DEEPSEEK_API_KEY or "BURAYA" in DEEPSEEK_API_KEY: 
    st.error("DeepSeek Key Eksik!"); st.stop()

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# 2. Google Gemini Ayarı (GÖZLER)
if GOOGLE_API_KEY and "BURAYA" not in GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ Google API Key girilmediği için Görsel Tanıma çalışmayacak.")

st.title("💊 AI Eczacı Asistanı (Sessiz Mod)")
st.caption("Beyin: DeepSeek-V3 | Göz: Gemini 1.5/2.0 Flash | Kaynak: PDF & FDA")

# =============================================================================
# 2. RAG MOTORU
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
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 10]
        for i in range(0, len(sentences), 5):
            chunk = " ".join(sentences[i : i + 15])
            self.vector_store.append({
                "source": source_name,
                "content": f"[{source_name}] {chunk}",
                "vector": self.get_embedding(chunk)
            })
        if source_name not in self.loaded_files: self.loaded_files.append(source_name)

    def retrieve(self, query, top_k=5): 
        # 1. YÖNTEM: İSİM EŞLEŞTİRME (EN GARANTİ YOL)
        # Eğer kullanıcının sorusunda, yüklenen dosyanın ismi geçiyorsa (Örn: Soru "Parol nedir?", Dosya "PAROL")
        # Puan hesaplamasıyla vakit kaybetme, direkt o dosyanın içeriğini getir.
        query_upper = query.upper()
        priority_content = []
        
        for item in self.vector_store:
            # Dosya ismi (source) sorunun içinde geçiyor mu?
            if item["source"] in query_upper and len(item["source"]) > 2:
                priority_content.append(item["content"])
        
        # Eğer isimden yakaladıysak hemen döndür (En üstteki 3 parça yeterli)
        if priority_content:
            return "\n\n".join(priority_content[:3])

        # 2. YÖNTEM: VEKTÖR BENZERLİĞİ (YEDEK)
        query_vec = self.get_embedding(query)
        candidates = self.vector_store
        scores = []
        for item in candidates:
            score = self._cosine_similarity(query_vec, item["vector"])
            scores.append((score, item["content"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # --- DÜZELTME: EŞİK DEĞERİ DÜŞÜRÜLDÜ ---
        # 0.4 çok yüksekti, 0.15 yapıyoruz. SimpleRAG için ideal aralık budur.
        if not scores or scores[0][0] < 0.15: 
            return "VERİTABANINDA YOK"
            
        return "\n\n".join([item[1] for item in scores[:top_k]])

@st.cache_resource
def load_rag_system():
    rag = SimpleRAG()
    if not os.path.exists("./pdf_data"): os.makedirs("./pdf_data")
    for f in os.listdir("./pdf_data"):
        if f.endswith('.pdf'):
            try:
                reader = PdfReader(os.path.join("./pdf_data", f))
                text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
                rag.ingest(text, source_name=f.replace(".pdf", "").upper())
            except: pass
    return rag
rag_engine = load_rag_system()

# =============================================================================
# 3. YARDIMCI ARAÇLAR
# =============================================================================

# 📸 GÖRSEL ANALİZ
def analyze_image(image_bytes):
    # Hafızayı temizle (önceki ilaç karışmasın)
    if "last_fetched_content" in st.session_state:
        del st.session_state["last_fetched_content"]
        
    image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]
    prompt = "Bu resimdeki ilaç kutusunun üzerindeki MARKA İSMİNİ tespit et. Sadece ismi yaz. Başka hiçbir şey yazma."

    # Önce en hızlı ve güncel modeli dener
    models = ['gemini-flash-latest', 'gemini-1.5-flash', 'gemini-2.0-flash-lite-preview-02-05']

    for m in models:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content([prompt, image_parts[0]])
            text = response.text.strip()
            if text: return text
        except Exception:
            time.sleep(1) 
            continue

    return "❌ Okunamadı"

def clean_drug_name(text):
    text = re.split(r"['’:]", text)[0] 
    stopwords = ["fotoğrafta", "tespit", "edilen", "ilaç", "nedir", "ne", "işe", "yarar", "yan", "etkileri", "kullanımı", "hakkında", "bilgi", "fiyatı"]
    words = [w for w in text.split() if w.lower() not in stopwords and len(w) > 2]
    return words[0] if words else text.split()[0]

def fetch_drug_from_api(input_text):
    drug_name = clean_drug_name(input_text)
    if not drug_name or len(drug_name) < 2 or "BULUNAMADI" in drug_name: return "❌ İlaç ismi anlaşılamadı."
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{drug_name}\"&limit=1"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                result = data["results"][0]
                info_parts = []
                fields = {"indications_and_usage": "KULLANIM", "warnings": "UYARILAR", "adverse_reactions": "YAN ETKİLER", "boxed_warning": "KRİTİK UYARI"}
                for key, label in fields.items():
                    if key in result:
                        content = result[key]
                        if isinstance(content, list): content = " ".join(content)
                        info_parts.append(f"--- {label} ---\n{content[:2000]}")
                full_text = "\n\n".join(info_parts)
                if len(full_text) < 50: return f"❌ '{drug_name}' boş kayıt."
                st.session_state["last_fetched_content"] = full_text
                rag_engine.ingest(full_text, source_name=f"FDA_{drug_name.upper()}")
                return f"✅ BAŞARILI: '{drug_name}' verisi çekildi."
            else: return f"❌ '{drug_name}' bulunamadı."
        else: return f"❌ API Hatası: {response.status_code}"
    except Exception as e: return f"⚠️ Hata: {str(e)}"

# =============================================================================
# 4. AGENTLER (DEEPSEEK)
# =============================================================================

# 🧠 YÖNETİCİ AJAN (TAM GÜVENLİK MODU - STRICT MODE)
def manager_agent(user_input):
    with st.status("🚀 İşlem Yapılıyor...", expanded=True) as status:
        
        # 1. ADIM: YEREL RAG (PDF)
        status.write("📂 **Adım 1:** Yerel PDF Arşivi Taranıyor...")
        rag_context = rag_engine.retrieve(user_input)
        
        source_type = "YEREL DOKÜMAN"
        found_data = False # Veri bulup bulmadığımızı takip eden bayrak

        # Eğer PDF'te yoksa (Skor düşükse veya boşsa)
        if rag_context == "VERİTABANINDA YOK":
            status.write("🌍 **Adım 2:** Yerel arşivde yok, Global FDA veritabanına bağlanılıyor...")
            
            drug_name = clean_drug_name(user_input)
            
            # İlaç ismi tespit edilebildiyse API'ye sor
            if len(drug_name) > 2:
                fetch_result = fetch_drug_from_api(drug_name)
                
                if "✅" in fetch_result:
                    status.write(f"📥 **Bulundu:** {fetch_result}")
                    rag_context = st.session_state.get("last_fetched_content", "")
                    source_type = "FDA RESMİ KAYIT"
                    found_data = True
                else:
                    status.write("❌ **Sonuç:** FDA kayıtlarında da bulunamadı.")
                    found_data = False
            else:
                status.write("⚠️ İlaç ismi net anlaşılamadı.")
                found_data = False
        else:
            status.write("✅ **Bulundu:** Yerel arşivden eşleşme sağlandı.")
            found_data = True

        # Eğer ne PDF'te ne de FDA'da veri yoksa, DeepSeek'e hiç gitme!
        if not found_data:
            status.update(label="❌ Veri Bulunamadı", state="error", expanded=False)
            return f"❌ **Üzgünüm, aradığınız '{user_input}' hakkında veritabanımda bilgi bulunmamaktadır.**\n\nGüvenlik protokolleri gereği, yerel veritabanında (PDF) veya uluslararası resmi kayıtlarda (FDA) doğrulanmamış ilaçlar hakkında yorum yapamıyorum.\n\nLütfen ilaç ismini kontrol edip tekrar deneyiniz veya bir doktora danışınız."

        # 3. ADIM: CEVAP ÜRETİMİ (Sadece Veri Varsa Çalışır)
        status.write("🧠 **Adım 3:** Eczacı Asistanı yanıtlıyor...")
        
        system_prompt = f"""
        Sen Uzman bir Eczacı Asistanısın.
        KULLANILAN KAYNAK: {source_type}

        
        TALİMATLAR:
        1. SADECE sana verilen [DOKÜMANLAR] kısmındaki bilgiyi kullanarak cevap ver.
        2. Asla dışarıdan bilgi ekleme veya uydurma.
        3. CEVAPLARIN KISA, NET VE DOĞRUDAN OLSUN.
        4. Gereksiz giriş/çıkış cümleleri ("Size yardımcı olayım", "Özetle" vb.) ASLA kullanma.
        5. Cevabı maksimum 4-5 cümle ile sınırla.
        6. Eğer soru "Ne işe yarar?" ise sadece endikasyonları say.
        
        """
        
        full_prompt = f"SORU: {user_input}\n\nDOKÜMANLAR:\n{rag_context}"
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0 # Sıfır yaratıcılık, tam itaat
            )
            final_answer = response.choices[0].message.content
            
            status.update(label="✅ Tamamlandı", state="complete", expanded=False)
            return final_answer
            
        except Exception as e:
            return f"⚠️ Yapay Zeka Hatası: {str(e)}"

# =============================================================================
# 5. ARAYÜZ (UI)
# =============================================================================
with st.sidebar:
    st.header("📸 Görsel Tanı (Gemini)")
    img_file = st.file_uploader("İlaç Fotoğrafı", type=["jpg", "png", "jpeg"])
    
    if img_file:
        if "processed_file_name" not in st.session_state or st.session_state["processed_file_name"] != img_file.name:
            with st.spinner("🤖 Gemini Fotoğrafı İnceliyor..."):
                detected_name = analyze_image(img_file.getvalue())
            
            if detected_name and "❌" not in detected_name:
                st.success(f"✅ İlaç: {detected_name}")
                st.session_state["auto_prompt"] = f"{detected_name} ilacı nedir, ne için kullanılır?"
                st.session_state["processed_file_name"] = img_file.name
                st.rerun()
            else: st.error("❌ İlaç ismi görselden okunamadı.")

    st.divider()
    st.header("📂 PDF Ekle")
    files = st.file_uploader("Veri Ekle", accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join("./pdf_data", f.name), "wb") as file: file.write(f.getbuffer())
        st.cache_resource.clear()
        st.toast("Eklendi!")

# Mesaj Geçmişi
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# Otomatik Soru (Görselden gelirse) veya Manuel Soru
if "auto_prompt" in st.session_state:
    user_prompt = st.session_state["auto_prompt"]
    del st.session_state["auto_prompt"]
else:
    user_prompt = st.chat_input("Soru sor...")

# Ana İşleyiş
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): st.markdown(user_prompt)
    
    with st.chat_message("assistant"):
        response = manager_agent(user_prompt)
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})