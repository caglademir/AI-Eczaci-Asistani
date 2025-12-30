💊 AI Eczacı Asistanı (HIB-RAG v3.6)
"Güvenli, Denetlenebilir ve Hibrit Mimarili İlaç Danışmanlık Sistemi"
Bu proje, Büyük Dil Modellerinin (LLM) sağlık alanında güvenli kullanımını sağlamak amacıyla geliştirilmiş, %100 Güvenlik (Safety) odaklı bir "Karar Destek Sistemidir" (CDSS). Standart RAG yapılarının ötesine geçerek, görsel algılama (Vision) ve deterministik güvenlik protokollerini birleştiren Custom ReAct mimarisini kullanır.

🚀 Projenin Öne Çıkan Özellikleri
	•	🧠 Hibrit RAG Mimarisi: Yerel PDF veritabanı ile global FDA API'sini birleştiren, veri yoksa cevap vermeyi reddeden güvenli yapı.
	•	👁️ Multimodal (Görsel) Analiz: Google Gemini 1.5 Flash entegrasyonu ile ilaç kutularının fotoğrafından ilaç tespiti (OCR + Context).
	•	🛡️ Strict Safety Mode: Ölümcül ilaç etkileşimlerini ve yanlış kullanım senaryolarını yakalayan, halüsinasyon görmeyen (Non-Hallucinating) katı güvenlik protokolleri.
	•	📉 Maliyet Optimizasyonu: OpenAI GPT-4o modelinden DeepSeek-V3 modeline geçiş yapılarak %95 oranında maliyet tasarrufu sağlanmıştır.
	◦	GPT-4o Giriş Maliyeti: ~$2.50 / 1M Token
	◦	DeepSeek-V3 Giriş Maliyeti: ~$0.14 / 1M Token

🏗️ Sistem Mimarisi ve Teknik Detaylar
Sistem, app.py üzerinde çalışan özelleştirilmiş bir ReAct (Reasoning + Acting) ajanı üzerine kuruludur.
1. Akış Diyagramı (ReAct Loop)
	1	Gözlem (Observation): Kullanıcı metin mi yazdı, fotoğraf mı yükledi?
	2	Multimodal İşleme: Fotoğraf varsa Gemini 1.5 ile analiz et -> Metne dök.
	3	Aksiyon 1 (Primary Retrieval): CustomSimpleRAG ile yerel PDF'leri tara.
	◦	Özelleştirilmiş Chunking: Cümle bütünlüğünü koruyan sentences[i:i+15] örtüşmeli bölümleme.
	4	Karar (Decision Node):
	◦	Veri Yetersizse ➔ Aksiyon 2: FDA API'sine bağlan.
	◦	Veri Yoksa ➔ STOP: "Veritabanında Bulunamadı" hatası dön (Halüsinasyon Engeli).
	5	Final Sentez: DeepSeek-V3 Reasoning motoru ile yanıt üret.

📊 Benchmark ve Performans Sonuçları
Sistem, 40 soruluk "Adversarial Stress Test" senaryolarında test edilmiştir. Değerlendirme 5-Katmanlı Hakem Mimarisi (Cosine, ROUGE, BERTScore, Keyword, Entity) ile yapılmıştır.
Metrik
Başarı Oranı
Açıklama
Güvenlik (Safety)
%100
Kritik/Ölümcül senaryoların tamamında "HAYIR/SAKIN" uyarısı verildi.
Halüsinasyon Reddi
%100
Kriptonit veya Ferrari 500mg gibi uydurma ilaçlar reddedildi.
Tıbbi Terminoloji
Yüksek (0.92)
Laktik Asidoz, Anafilaksi gibi terimler doğru bağlamda kullanıldı.

Vaka Analizleri (Success & False Negatives)
	•	✅ Başarılı Vaka (Viagra + Nitrat): Sistem bu kombinasyonu "ÖLÜMCÜL RİSK" olarak işaretlemiş, tansiyon düşüklüğü mekanizmasını açıklamış ve kullanıcıyı acile yönlendirmiştir.
	•	⚠️ False Negative (Augmentin Kırma): Model "Sakın kırmayın, etkisi bozulur" diyerek doğru cevap vermesine rağmen, testteki yasaklı kelime filtresine ("kır") takılarak puan kaybetmiştir. Bu, sistemin "Fail-Safe" (Aşırı Güvenli) çalıştığını kanıtlar.

🛠️ Kurulum ve Çalıştırma
Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.
Gereksinimler
	•	Python 3.10 veya üzeri
	•	DeepSeek API Anahtarı
	•	Google Gemini API Anahtarı
Adım 1: Repoyu Klonlayın
Bash

git clone https://github.com/kullaniciadi/ai-eczaci-asistani.git
cd ai-eczaci-asistani
Adım 2: Kütüphaneleri Yükleyin
Bash

pip install -r requirements.txt
Adım 3: .env Dosyasını Oluşturun
Ana dizinde .env dosyası oluşturun ve anahtarlarınızı ekleyin:
Kod snippet'i

DEEPSEEK_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
Adım 4: Uygulamayı Başlatın
Bash

streamlit run app.py

📂 Proje Yapısı
ai-eczaci-asistani/
├── app.py                   # Ana uygulama (Streamlit + ReAct Agent)
├── benchmark_ultimate.py    # 5-Katmanlı Test Motoru
├── pdf_data/                # İlaç prospektüsleri (PDF)
├── DeepSeek_Benchmark_Raporu.xlsx # Detaylı test sonuçları
├── requirements.txt         # Bağımlılıklar
└── README.md                # Dokümantasyon

🔮 Gelecek Çalışmalar
	•	Vektör Veritabanı: RAM tabanlı yapıdan Pinecone veya ChromaDB'ye geçiş.
	•	Session Memory: Hastanın geçmiş ilaç kullanımını hatırlayan anamnez modülü.
	•	Sesli Asistan: Whisper entegrasyonu ile sesli komut özelliği (Beta aşamasında kaldırıldı, tekrar eklenecek).

⚠️ Yasal Uyarı (Disclaimer)
Bu proje bir tıbbi tavsiye aracı değildir. Sistem, eğitim ve araştırma amaçlı geliştirilmiş bir "Karar Destek Mekanizmasıdır". Üretilen bilgilerin doğruluğu %100 garanti edilmez. Sağlık sorunlarınızda mutlaka bir doktora veya eczacıya danışınız.

Geliştirici: Çağla Demir
Tarih: Aralık 2025 Lisans: MIT
