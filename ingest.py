import os
import requests
import json
import uuid

CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN")
INDEX_NAME = "wx-kb" 

AI_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/baai/bge-m3"
VECTOR_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/vectorize/v2/indexes/{INDEX_NAME}/insert"

HEADERS_AI = {"Authorization": f"Bearer {CF_API_TOKEN}"}
HEADERS_VECTOR = {
    "Authorization": f"Bearer {CF_API_TOKEN}",
    "Content-Type": "application/x-ndjson"
}

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def get_embeddings(texts):
    res = requests.post(AI_URL, headers=HEADERS_AI, json={"text": texts})
    return res.json()["result"]["data"]

def load_processed_files(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()

def mark_as_processed(log_path, filename):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

def ingest_directory(directory_path):
    log_file_path = os.path.join(directory_path, ".ingested.txt")
    processed_files = load_processed_files(log_file_path)
    
    for filename in os.listdir(directory_path):
        if not filename.endswith(".md"):
            continue
            
        if filename in processed_files:
            print(f"⏩跳过已处理文件: {filename}")
            continue
            
        print(f"⚙️正在处理: {filename}")
        with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
            content = f.read()
            
        chunks = chunk_text(content)
        
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch_texts = chunks[i:i+batch_size]
            embeddings = get_embeddings(batch_texts)
            
            vectors_to_insert = []
            for j, emb in enumerate(embeddings):
                vectors_to_insert.append({
                    "id": str(uuid.uuid4()), 
                    "values": emb,
                    "metadata": {"text": batch_texts[j]} 
                })
                
            lines = "\n".join([json.dumps(v) for v in vectors_to_insert])
            
            res = requests.post(VECTOR_URL, headers=HEADERS_VECTOR, data=lines)
            
            if res.status_code == 200:
                pass
            else:
                print(f"❌批量插入失败，状态码: {res.status_code}，响应: {res.text}")
                return 
            
        print(f"✅完成入库: {filename}")
        mark_as_processed(log_file_path, filename)

if __name__ == "__main__":
    ingest_directory("/data")
