# 🤖 Offline RAG AI Assistant with Microsoft Foundry Local

Bu proje, **Microsoft Foundry Local SDK** altyapısını ve **RAG (Retrieval-Augmented Generation)** mimarisini kullanarak, tamamen yerel (çevrimdışı/offline) çalışan gelişmiş bir doküman soru-cevap asistanıdır.

Dış API'lere veya aktif bir internet bağlantısına ihtiyaç duymadan, hassas verilerinizi bilgisayarınızdan dışarı çıkarmadan belgeleriniz üzerinden akıllı soru-cevap yapmanızı sağlar.

---

## 🌟 Öne Çıkan Özellikler

- **🔒 %100 Çevrimdışı ve Güvenli:** Verileriniz internete çıkmaz, tüm vektörleştirme ve LLM çıkarımı yerel donanımda gerçekleşir.
- **📄 Çoklu Format Desteği:** PDF ve TXT formatındaki belgeleri otomatik olarak işler ve anlamlı metin parçalarına (chunks) böler.
- **⚡ Hafif & Hızlı Vektör Veritabanı:** Metin parçaları ve bunlara ait embedding (vektör) verileri yerel **SQLite** veritabanında saklanır.
- **🎯 Cosine Similarity Araması:** Kullanıcı sorusu ile en alakalı doküman parçalarını belirlemek için matematiksel vektör benzerliği hesaplar.
- **🛡️ Hallucination (Yanlış Bilgi) Önleme:** Özel tasarlanmış sistem promptu sayesinde asistan, belgelerde olmayan bilgileri uydurmaz ve yanıtlarında kaynak gösterir.

---

## 🛠️ Teknolojik Mimari (Tech Stack)

| Bileşen | Kullanılan Teknoloji / Kütüphane | Açıklama |
| :--- | :--- | :--- |
| **LLM Runtime** | Microsoft Foundry Local SDK (`phi-3.5-mini`) | Yerel dil modeli çıkarım motoru |
| **Embedding Modeli** | `paraphrase-multilingual-MiniLM-L12-v2` | Türkçe ve çok dilli metin vektörleştirme |
| **Vektör Veritabanı**| SQLite3 | Metin parçaları ve binary vektör depolama |
| **PDF Okuyucu** | `pypdf` | PDF belgelerinden metin çıkarma |
| **Text Okuyucu** | Python `open()` (Built-in) | Düz metin (.txt) dosyalarını okuma |
| **Matematik & Vektör**| `numpy` | Kosinüs benzerliği (Cosine Similarity) hesaplamaları |

---

## 📐 Çalışma Mantığı (Pipeline)

```text
[ Dokümanlar (.pdf / .txt) ] 
            │
            ▼
   1. Chunking (Parçalama)
            │
            ▼
   2. Embedding (SentenceTransformers)
            │
            ▼
   3. SQLite Veritabanı (Vektör Saklama)
            │
  ┌─────────┴─────────┐
  │  Kullanıcı Sorusu │ ──> Vektörleştirme
  └─────────┬─────────┘
            │
            ▼
   4. Cosine Similarity Araması (top_k=1)
            │
            ▼
   5. Microsoft Foundry Local (phi-3.5-mini)
            │
            ▼
   [ Doğru ve Kaynaklı Türkçe Yanıt ]
