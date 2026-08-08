import os
import sqlite3
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. MODEL VE EMBEDDING BAŞLATMA (Phase 1)
print("⏳ Türkçe destekli embedding modeli yükleniyor...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

client = OpenAI(
    base_url="http://localhost:5272/v1",
    api_key="foundry"
)

DB_PATH = "database/rag_proje.db"

# 2. VERİTABANI MİMARİSİ
def veritabani_kur():
    if not os.path.exists("database"): 
        os.makedirs("database")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS belgeler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dosya_adi TEXT,
            parca_no INTEGER,
            icerik TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

# 3. DATA INGESTION & CHUNKING
def veri_isle_if_needed():
    veritabani_kur()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM belgeler")
    count = cursor.fetchone()[0]
    if count > 0:
        print(f"ℹ️ Veritabanında {count} adet kayıtlı metin parçası mevcut. İndeksleme atlanıyor.")
        conn.close()
        return

    if not os.path.exists("data"):
        os.makedirs("data")
        
    dosyalar = os.listdir("data")
    if not dosyalar:
        print("⚠️ 'data' klasöründe okunacak dosya bulunamadı! Lütfen 'data/' klasörüne PDF veya TXT ekleyin.")
        conn.close()
        return

    for dosya in dosyalar:
        yol = os.path.join("data", dosya)
        print(f"📄 Vektörleştiriliyor: {dosya}")
        
        ham_metin = ""
        if dosya.endswith('.pdf'):
            reader = PdfReader(yol)
            for p in reader.pages:
                txt = p.extract_text()
                if txt: ham_metin += txt + "\n\n"
        elif dosya.endswith('.txt'):
            with open(yol, 'r', encoding='utf-8') as f:
                ham_metin = f.read()

        parcalar = [p.strip() for p in ham_metin.split("\n\n") if p.strip()]

        for idx, parca in enumerate(parcalar):
            vec = embedder.encode(parca).astype(np.float32).tobytes()
            cursor.execute("INSERT INTO belgeler (dosya_adi, parca_no, icerik, embedding) VALUES (?, ?, ?, ?)",
                           (dosya, idx + 1, parca, vec))
            
    conn.commit()
    print("✅ Tüm dokümanlar başarıyla indekslendi.")
    conn.close()

# 4. EN ALAKALI TEK VEYA İKİ PARÇAYI GETİRME (top_k=1, esik_skor=0.50)
def get_relevant_chunks(soru, top_k=1, esik_skor=0.50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT dosya_adi, icerik, embedding FROM belgeler")
    kayitlar = cursor.fetchall()
    conn.close()
    
    if not kayitlar:
        return []

    q_vec = embedder.encode(soru).astype(np.float32)
    skorlar = []

    for dosya_adi, icerik, emb_blob in kayitlar:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = np.dot(q_vec, emb) / (np.linalg.norm(q_vec) * np.linalg.norm(emb))
        
        if sim >= esik_skor:
            skorlar.append((sim, dosya_adi, icerik))

    skorlar.sort(key=lambda x: x[0], reverse=True)
    return skorlar[:top_k]

# 5. RAG PIPELINE
def asistan(soru):
    # En alakalı SADECE 1 parçayı getiriyoruz (top_k=1)
    en_alakali_parcalar = get_relevant_chunks(soru, top_k=1, esik_skor=0.50)
    
    if not en_alakali_parcalar:
        return "⚠️ Üzgünüm, sağlanan belgelerde bu soruyla ilgili yeterli bilgi bulunamadı."

    baglam_metni = ""
    for idx, (skor, dosya, icerik) in enumerate(en_alakali_parcalar, 1):
        baglam_metni += f"\n--- Kaynak Dosya: {dosya} (Benzerlik Skoru: {skor:.2f}) ---\n{icerik}\n"

    system_prompt = (
        "Sen dürüst bir Asistansın. Sana verilen bağlamı (Context) dikkatlice oku ve kullanıcının sorusuna cevap ver.\n"
        "ÖNEMLİ KURALLAR:\n"
        "1. Sadece verilen bağlamdaki bilgileri kullan, dışarıdan bilgi uydurma.\n"
        "2. Eğer verilen bağlamda cevap yoksa, kesinlikle tahmin yürütme ve 'Bu bilgi belgelerimde yer almıyor' de.\n"
        "3. Cevap verirken hangi dosyadan (Kaynak Dosya) bilgi aldığını mutlaka belirt."
    )

    user_prompt = f"Bağlam:\n{baglam_metni}\n\nSoru: {soru}\nCevap:"

    try:
        res = client.chat.completions.create(
            model="phi-3.5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return res.choices[0].message.content
    except Exception:
        return f"ℹ️ [Local LLM Bağlantısı Bekleniyor - Bulunan En Alakalı Bağlam]:\n{baglam_metni}"

if __name__ == "__main__":
    veri_isle_if_needed()
    
    print("\n-------------------------------------------------")
    print("🤖 Microsoft Foundry Local - Çevrimdışı RAG Asistanı")
    print("-------------------------------------------------")
    print("Çıkmak için 'q' yazabilirsiniz.\n")
    
    while True:
        soru = input("\nSoru: ")
        if soru.lower() in ['exit', 'çıkış', 'q']: 
            break
        print("\n🤖 Cevap:\n", asistan(soru))