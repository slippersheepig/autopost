'''
This is the main Python file that runs the Flask application and the LLM buffering proxy with session history.
'''
import os
import requests
import random
import threading
from flask import Flask, render_template, request, jsonify
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

# ==========================================
# 环境变量配置
# ==========================================
API_KEY = os.environ.get("API_KEY", "your_api_key")
API_URL = os.environ.get("API_URL", "https://example.com")
MODEL = os.environ.get("MODEL", "@cf/meta/llama-3.2-11b-vision-instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "10"))

# 线程锁，确保多用户并发访问字典时的绝对安全
pool_lock = threading.Lock()
# 任务池：仅存放当前正在异步计算的瞬时状态 {"user_id": {"status": "processing", "result": ""}}
task_pool = {}
# 历史记录池：持久存放每个用户的上下文历史 {"user_id": [{"role": "user", "content": "..."}, ...]}
history_pool = {}

def fetch_llm(user_id, prompt):
    """后台执行的大模型请求逻辑，自动组装历史上下文并动态截断"""
    global task_pool, history_pool
    
    with pool_lock:
        # 如果是新用户，初始化其历史队列
        if user_id not in history_pool:
            history_pool[user_id] = []
            
        # 将当前新问题追加进历史
        history_pool[user_id].append({"role": "user", "content": prompt})
        
        # 核心逻辑：合理限制历史记录长度，只保留最近的 N 条记录
        if len(history_pool[user_id]) > MAX_HISTORY:
            history_pool[user_id] = history_pool[user_id][-MAX_HISTORY:]
            
        # 复制一份当前的完整上下文用于发送请求，避免后台请求期间发生线程读写冲突
        current_messages = list(history_pool[user_id])
        
        # 初始化当前单次任务的取件状态
        task_pool[user_id] = {"status": "processing", "result": ""}
    
    payload = {
        "model": MODEL,
        "messages": current_messages,
        "max_tokens": MAX_TOKENS
    }
    
    try:
        res = requests.post(
            API_URL, 
            json=payload, 
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=120
        )
        data = res.json()
        response_text = data["choices"][0]["message"]["content"]
        
        with pool_lock:
            # 将模型的回答也追加进该用户的历史记录中
            history_pool[user_id].append({"role": "assistant", "content": response_text})
            
            # 再次截断确保历史长度合规
            if len(history_pool[user_id]) > MAX_HISTORY:
                history_pool[user_id] = history_pool[user_id][-MAX_HISTORY:]
                
            # 更新单次任务状态，供用户“取件”
            task_pool[user_id]["result"] = response_text
            task_pool[user_id]["status"] = "done"
            
    except Exception as e:
        with pool_lock:
            task_pool[user_id]["result"] = f"请求出错: {str(e)}"
            task_pool[user_id]["status"] = "error"
            # 容错：如果请求失败，把刚放进去的问题弹出来，避免污染上下文
            if history_pool[user_id] and history_pool[user_id][-1]["role"] == "user":
                history_pool[user_id].pop()

# ==========================================
# 极简鉴权拦截器
# ==========================================
def verify_auth():
    """验证请求头中的 Authorization 是否与本地 API_KEY 匹配"""
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {API_KEY}":
        return False
    return True

# ==========================================
# AI 缓冲代理 API 路由
# ==========================================
@app.route('/ask', methods=['POST'])
def ask_question():
    if not verify_auth():
        return jsonify({"msg": "error", "error": "Unauthorized"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"msg": "error", "error": "Invalid JSON"}), 400
        
    user_id = data.get("user_id")
    text = data.get("text")
    
    if not user_id or not text:
        return jsonify({"msg": "error", "error": "Missing user_id or text"}), 400

    thread = threading.Thread(target=fetch_llm, args=(user_id, text))
    thread.start()
    
    return jsonify({"msg": "ok"})

@app.route('/get_result', methods=['GET'])
def get_result():
    if not verify_auth():
        return jsonify({"status": "error", "data": "Unauthorized"}), 401

    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "data": "Missing user_id"})
        
    with pool_lock:
        if user_id not in task_pool:
            return jsonify({"status": "idle", "data": "当前没有正在处理的问题，请先发送 ai+问题 进行提问。"})
            
        task = task_pool[user_id]
        if task["status"] in ["done", "error"]:
            res = task["result"]
            del task_pool[user_id]
            return jsonify({"status": "done", "data": res})
        elif task["status"] == "processing":
            return jsonify({"status": "processing", "data": "模型还在疯狂输出中，请稍后再次发送『取件』..."})

@app.route('/clear', methods=['POST'])
def clear_history():
    if not verify_auth():
        return jsonify({"msg": "error", "error": "Unauthorized"}), 401

    data = request.get_json()
    user_id = data.get("user_id") if data else None
    if not user_id:
        return jsonify({"msg": "error", "error": "Missing user_id"}), 400
        
    with pool_lock:
        if user_id in history_pool:
            del history_pool[user_id]
        if user_id in task_pool:
            del task_pool[user_id]
            
    return jsonify({"msg": "ok"})

# ==========================================
# 原有的 ASCII Bear 路由
# ==========================================
@app.route('/')
def index():
    return render_template('index.html', random=random)

@app.route('/bear.txt')
def bear():
    random_color = '#%06x' % random.randint(0, 0xFFFFFF)
    return render_template('bear.txt', random_color=random_color)

if __name__ == '__main__':
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    print("Server started on port 5000...")
    http_server.serve_forever()
