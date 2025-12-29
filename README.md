# 💊 AI Eczacı Asistanı Pro (v3.0)

Bu proje, ilaç kutularını fotoğraftan tanıyan, prospektüsleri analiz eden ve global veritabanlarından (FDA) ilaç bilgisi çeken yapay zeka destekli bir eczacı asistanıdır.

## 🚀 Özellikler

* **📸 Fotoğraftan Tanıma:** İlaç kutusunun fotoğrafını yükleyin, yapay zeka ismini okusun.
* **🌍 Global Veritabanı:** Yerel veritabanında olmayan ilaçlar için otomatik olarak OpenFDA (Amerikan İlaç Dairesi) API'sine bağlanır.
* **🧠 RAG (Retrieval-Augmented Generation):** Yüklenen PDF prospektüsleri içinde akıllı arama yapar.
* **🇹🇷 Otomatik Çeviri:** Yabancı kaynaklardan gelen verileri anlık olarak Türkçe'ye çevirir.
* **🗣️ Sesli Yanıt:** Cevapları sesli olarak (Text-to-Speech) okur.
* **🤖 Çoklu Ajan Mimarisi:** Yönetici ve Araştırmacı ajanlar iş bölümü yaparak çalışır.

## 📂 Proje Yapısı

* `app.py`: Uygulamanın ana dosyası (Streamlit arayüzü ve tüm mantık).
* `main.py`: Komut satırı (CLI) üzerinden çalışan, arayüzsüz prototip versiyonu.
* `pdf_data/`: İlaç prospektüslerinin (PDF) saklandığı klasör.

## 🛠️ Kurulum

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Gerekli Kütüphaneleri Yükleyin:**
    Terminal veya komut satırına şu kodu yazın:
    ```bash
    pip install streamlit openai pypdf requests beautifulsoup4
    ```

2.  **API Anahtarını Ayarlayın:**
    `app.py` dosyasını açın ve `OPENAI_API_KEY` değişkenine kendi OpenAI API anahtarınızı yapıştırın.

3.  **Uygulamayı Başlatın:**
    Terminalde proje klasörüne gidip şu komutu çalıştırın:
    ```bash
    streamlit run app.py
    ```

## 💡 Nasıl Kullanılır?

1.  **Fotoğraf ile:** Sol panelden "Fotoğraf ile Tanı" kısmına ilaç kutusunun fotoğrafını sürükleyin. Sistem ilacı tanıyıp otomatik bilgi verecektir.
2.  **Metin ile:** Sohbet kutusuna "Prozac yan etkileri nelerdir?" gibi sorular sorun.
3.  **PDF Ekleme:** Elinizde özel bir ilaç PDF'i varsa sol panelden sisteme yükleyin, veritabanına eklensin.

## 🔧 Kullanılan Teknolojiler

* **Frontend:** Streamlit
* **LLM:** OpenAI GPT-4o
* **Vision:** GPT-4o Vision
* **Data Source:** OpenFDA API & Local PDFs
* **Audio:** OpenAI TTS-1

---
**Geliştirici:** Çağla DEMİR 2020556018
**Tarih:** 2025