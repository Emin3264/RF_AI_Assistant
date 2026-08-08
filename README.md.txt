# 🤖 Offline RAG AI Assistant with Microsoft Foundry Local

Bu proje, **Microsoft Foundry Local** mimarisini ve **RAG (Retrieval-Augmented Generation)** desenini kullanarak, tamamen yerel (çevrimdışı/offline) çalışan bir doküman soru-cevap asistanıdır.

## 🌟 Öne Çıkan Özellikler
- **Tamamen Çevrimdışı (Offline):** İnternet bağlantısı veya dış API'lere bağımlılık duymadan çalışır.
- **Doküman İşleme & Parçalama (Chunking):** PDF ve TXT formatındaki belgeleri okur ve paragraflara böler.
- **SQLite Vektör Veritabanı:** Metin parçalarını ve bunların binary (BLOB) vektör temsilcilerini SQLite üzerinde saklar.
- **Vektör Benzerlik Araması:** Kullanıcı sorusu ile doküman parçaları arasındaki Cosine Similarity (Kosinüs Benzerliği) değerini hesaplayarak en alakalı parçayı (`top_k=1`) getirir.
- **Prompt Engineering & Hallucination Önleme:** Bilgi uydurmayı (hallucination) engelleyen ve kaynak gösteren özel sistem promptu kullanır.

## 🛠️ Kullanılan Teknolojiler
- **Dil:** Python 3.10+
- **LLM Runtime:** Microsoft Foundry Local SDK (`phi-3.5-mini`)
- **Embedding Modeli:** `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers)
- **Veritabanı:** SQLite3
- **PDF Okuyucu:** `pypdf`

## 🚀 Kurulum ve Çalıştırma

1. **Gerekli Kütüphaneleri Yükleyin:**
   ```bash
   pip install sentence-transformers pypdf numpy openai foundry-local-sdk