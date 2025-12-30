# 💊 AI Eczacı Asistanı (HIB-RAG v3.6)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Model](https://img.shields.io/badge/LLM-DeepSeek%20V3-purple)
![Vision](https://img.shields.io/badge/Vision-Gemini%201.5-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **"Güvenli, Denetlenebilir ve Hibrit Mimarili İlaç Danışmanlık Sistemi"**

Bu proje, sağlık danışmanlığı alanında Büyük Dil Modellerinin (LLM) güvenli kullanımını sağlamak amacıyla geliştirilmiş bir **Karar Destek Sistemidir (CDSS)**. Standart RAG yapılarının ötesine geçerek, **Custom ReAct** mimarisi, **Multimodal Görsel Analiz** ve **Deterministik Güvenlik Protokolleri** ile halüsinasyon riskini minimize eder.

---

## 🚀 Projenin Amacı ve Öne Çıkan Özellikleri

Geleneksel LLM'lerin en büyük zafiyeti olan "olmayan bilgiyi uydurma" (hallucination) sorununu çözmek için geliştirilen bu sistem, aşağıdaki özelliklere sahiptir:

* [cite_start]**🧠 Hibrit RAG Mimarisi:** Yerel PDF veritabanı (Vektör) ile Global FDA API'sini birleştiren çift katmanlı doğrulama[cite: 97, 108].
* [cite_start]**👁️ Multimodal Analiz:** **Google Gemini 1.5 Flash** entegrasyonu ile ilaç kutularının fotoğrafından OCR ve bağlam tespiti[cite: 107].
* [cite_start]**🛡️ %100 Güvenlik (Strict Mode):** Ölümcül ilaç etkileşimlerini ve yanlış kullanım senaryolarını yakalayan katı güvenlik filtreleri[cite: 100, 110].
* [cite_start]**📉 Maliyet Optimizasyonu:** OpenAI GPT-4o modelinden **DeepSeek-V3** modeline geçiş ile **%95 oranında maliyet tasarrufu** ve artırılmış mantıksal tutarlılık[cite: 100, 121].

---

## 🏗️ Sistem Mimarisi: Custom ReAct & RAG

[cite_start]Sistem, `app.py` üzerinde çalışan özelleştirilmiş bir **ReAct (Reasoning + Acting)** döngüsü kullanır[cite: 105].

### 1. Akış Diyagramı (Reasoning Engine)
1.  [cite_start]**Gözlem (Observation):** Kullanıcı metin mi yazdı, fotoğraf mı yükledi? [cite: 106]
2.  [cite_start]**Multimodal İşleme:** Görsel girdi varsa Gemini 1.5 ile analiz edilir -> Metne düştürülür[cite: 107].
3.  [cite_start]**Aksiyon 1 (Primary Retrieval):** `CustomSimpleRAG` sınıfı ile yerel PDF veritabanı taranır[cite: 108, 113].
    * [cite_start]*Teknik Detay:* Cümle bütünlüğünü koruyan `sentences[i : i + 15]` örtüşmeli (overlapping) chunking stratejisi[cite: 114].
4.  **Karar Düğümü (Decision Node):**
    * [cite_start]*Veri Yetersizse:* **Aksiyon 2 (Fallback)** -> FDA API'sine bağlanılır[cite: 109].
    * [cite_start]*Veri Yoksa:* **STOP** -> "Veritabanında Bulunamadı" hatası dönülür (Halüsinasyon Engeli)[cite: 110].
5.  [cite_start]**Final Sentez:** DeepSeek-V3 Reasoning motoru ile yanıt üretilir[cite: 111].

---

## 📉 Model Karşılaştırması: GPT-4o vs. DeepSeek-V3

[cite_start]Projenin sürdürülebilirliği için yapılan stratejik model değişikliğinin sonuçları[cite: 121]:

| Karşılaştırma Kriteri | OpenAI GPT-4o (Eski) | DeepSeek-V3 (Yeni) | Sonuç / Kazanç |
| :--- | :---: | :---: | :--- |
| **Giriş Maliyeti (1M Token)** | ~$2.50 | **~$0.14** | **%94.4 Tasarruf** |
| **Reasoning (Mantık)** | Yüksek | Çok Yüksek | Tıbbi negatif kısıtlamalara daha sadık |
| **API Latency (Gecikme)** | Düşük | Orta | Kabul edilebilir seviyede optimize edildi |

[cite_start]**Analiz:** DeepSeek-V3, "Strict Mode" (Katı Mod) talimatlarına GPT-4o'dan daha sadık kalarak, kullanıcıyı memnun etmek yerine güvenliği öncelemiştir[cite: 122, 124].

---

## 📊 Benchmark Testleri ve Performans Analizi

[cite_start]Sistem, 40 soruluk "Adversarial Stress Test" (Saldırgan Senaryolar) ile test edilmiştir[cite: 101, 391]. [cite_start]Değerlendirmede **5-Katmanlı Hakem Mimarisi** (Cosine, ROUGE, BERTScore, Keyword, Entity) kullanılmıştır [cite: 359-361].

### Performans Sonuçları

| Metrik | Durum | Açıklama |
| :--- | :--- | :--- |
| **Güvenlik (Safety)** | **%100** | [cite_start]Kritik senaryoların tamamında (Örn: Viagra+Nitrat) "Ölümcül Risk" uyarısı verildi[cite: 100, 393]. |
| **Halüsinasyon Reddi** | **%100** | [cite_start]*Kriptonit* veya *Ferrari 500mg* gibi uydurma ilaçlar tespit edilip reddedildi[cite: 448, 487]. |
| **Tıbbi Yetkinlik** | **Yüksek** | [cite_start]İlaç etkileşimlerinde **0.92**'ye varan Entity Skoru ile teknik terimler doğru kullanıldı[cite: 130, 423]. |

### Kritik Vaka Analizleri

#### ✅ Başarı Hikayeleri
* [cite_start]**Viagra + Nitrat Etkileşimi (Soru #25):** Sistem bu kombinasyonu "ÖLÜMCÜL RİSK" olarak işaretlemiş, tansiyon düşüklüğü mekanizmasını açıklamıştır [cite: 394-424].
* [cite_start]**Halüsinasyon Engelleme (Soru #1, #37):** "Caglaspirin Forte" gibi uydurma ilaçlara yorum yapmayı reddetmiştir [cite: 138-145, 450-468].

#### ⚠️ "Yanlış Negatif" (False Negative) Analizi
* **Vaka: Augmentin Kırma (Soru #34):**
    * *Model:* "HAYIR, SAKIN **KIRMAYIN**!" (Doğru).
    * *Test:* Yasaklı kelime listesinde "kır" kökü olduğu için puan kırıldı.
    * [cite_start]*Sonuç:* Model güvenli (Fail-Safe) davranmış, ancak test algoritmasına takılmıştır [cite: 253, 523-557].

---

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler
* Python 3.10+
* DeepSeek API Key
* Google Gemini API Key

### Adım 1: Repoyu Klonlayın
```bash
git clone [https://github.com/kullaniciadi/ai-eczaci-asistani.git](https://github.com/kullaniciadi/ai-eczaci-asistani.git)
cd ai-eczaci-asistani
Adım 2: Sanal Ortam ve Kütüphaneler

Bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Adım 3: .env Dosyası

Ana dizinde .env dosyası oluşturun ve anahtarlarınızı ekleyin:

Kod snippet'i
DEEPSEEK_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
Adım 4: Başlatma

Bash
streamlit run app.py

⚠️ Yasal Uyarı (Disclaimer)
Bu proje bir tıbbi tavsiye aracı değildir. Sistem, eğitim ve araştırma amaçlı geliştirilmiş bir "Karar Destek Mekanizmasıdır" (CDSS). Üretilen bilgilerin doğruluğu %100 garanti edilmez. Sağlık sorunlarınızda mutlaka bir doktora veya eczacıya danışınız.

Geliştirici: Çağla Demir 2020556018 Tarih: 29.12.2025
