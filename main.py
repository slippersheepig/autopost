'''
This is the main Python file that runs the Flask application and the LLM buffering proxy with session history.
'''
import os
import requests
import random
import uuid
import threading
from flask import Flask, render_template, request, jsonify
from gevent.pywsgi import WSGIServer
from duckduckgo_search import DDGS

app = Flask(__name__, static_folder='static', static_url_path='/static')
if not os.path.exists("static"):
    os.makedirs("static")

# ==========================================
# 环境变量配置
# ==========================================
API_KEY = os.environ.get("API_KEY", "your_api_key")
API_URL = os.environ.get("API_URL", "https://example.com")
MODEL = os.environ.get("MODEL", "@cf/meta/llama-3.2-11b-vision-instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "10"))
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN")
INDEX_NAME = "wx-kb"

# 线程锁，确保多用户并发访问字典时的绝对安全
pool_lock = threading.Lock()
# 任务池：仅存放当前正在异步计算的瞬时状态 {"user_id": {"status": "processing", "result": ""}}
task_pool = {}
# 历史记录池：持久存放每个用户的上下文历史 {"user_id": [{"role": "user", "content": "..."}, ...]}
history_pool = {}

def search_knowledge_base(query_text):
    """新增功能：从 Cloudflare Vectorize 搜索最相关的文本"""
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return ""
        
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    ai_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/baai/bge-m3"
    query_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/vectorize/v2/indexes/{INDEX_NAME}/query"
    
    try:
        # 1. 把用户的问题变成向量
        emb_res = requests.post(ai_url, headers=headers, json={"text": [query_text]}).json()
        query_vector = emb_res["result"]["data"][0]
        
        # 2. 去数据库搜最相似的 3 个段落，并要求返回 metadata（原文）
        search_res = requests.post(query_url, headers=headers, json={
            "vector": query_vector,
            "topK": 3,
            "returnValues": False,
            "returnMetadata": "all"
        }).json()
        
        # 3. 拼接搜到的文本
        contexts = [match["metadata"]["text"] for match in search_res["result"]["matches"]]
        return "\n\n---\n\n".join(contexts)
    except Exception as e:
        print(f"检索失败: {e}")
        return ""

# ==========================================
# 新增：联网搜索辅助函数
# ==========================================
def get_web_search_context(query):
    """使用 DuckDuckGo 获取全网最新信息"""
    try:
        with DDGS() as ddgs:
            # 搜索前 3 条网页结果
            results = list(ddgs.text(query, max_results=3))
            
            if not results:
                return "未找到相关的网络搜索结果。"
                
            context = ""
            for i, r in enumerate(results):
                context += f"[{i+1}] 标题：{r.get('title', '')}\n摘要：{r.get('body', '')}\n链接：{r.get('href', '')}\n\n"
            return context
    except Exception as e:
        print(f"DuckDuckGo搜索出错: {e}")
        return ""

# ==========================================
# 新增：意图分析路由器 (Intent Router)
# ==========================================
def analyze_intent(query):
    """
    调用极速小模型，判断用户意图是否需要联网。
    返回 True (需要联网) 或 False (查知识库/闲聊)
    """
    # 强制使用速度最快的 8B 模型作为路由，不消耗主模型算力
    router_model = "@cf/meta/llama-3.1-8b-instruct-fp8"
    ai_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{router_model}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    
    # 极其严苛的系统提示词，逼迫模型只输出 YES 或 NO
    router_prompt = (
        "你是一个意图识别引擎。你的任务是判断用户的提问是否需要通过互联网搜索引擎获取最新资讯、"
        "实时数据、或是超出常规知识库的广泛网络信息。\n"
        "如果需要联网搜索，请严格且仅输出一个词：YES。\n"
        "如果是常规问答、闲聊、或者你认为本地知识库可能包含的内容，请严格且仅输出一个词：NO。\n"
        "不要输出任何标点符号和其他解释。"
    )
    
    payload = {
        "messages": [
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": 5, # 极小 token，保证光速响应
        "temperature": 0.1 # 降到最低，保证输出的确定性
    }
    
    try:
        res = requests.post(ai_url, headers=headers, json=payload, timeout=5)
        if res.status_code == 200:
            decision = res.json()["result"]["response"].strip().upper()
            print(f"🚦意图路由判断结果: {decision} | 用户问题: {query}")
            return "YES" in decision
        return False
    except Exception as e:
        print(f"路由判断异常: {e}")
        return False # 异常情况下，默认走本地知识库或闲聊

# ==========================================
# 升级：完全自动化的 fetch_llm
# ==========================================
def fetch_llm(user_id, prompt):
    """后台任务：自动路由 + 自动组装 + 大模型生成"""
    global task_pool, history_pool
    
    # 1. 过第一道关卡：意图路由
    needs_search = analyze_intent(prompt)
    
    # 2. 根据路由结果，去不同的“仓库”进货
    if needs_search:
        reference_context = get_web_search_context(prompt)
        system_role_desc = (
            "你是一个微信公众号的智能助手。当前处于【全网实时搜索】模式。\n"
            "请严格参考以下【联网搜索资料】来解答用户的问题，并在回答末尾附上原文链接。\n"
            "如果资料与问题无关，请诚实说明未搜到相关信息。\n\n"
            f"【联网搜索资料】开始：\n{reference_context}\n【联网搜索资料】结束。"
        )
    else:
        reference_context = search_knowledge_base(prompt)
        system_role_desc = (
            "你是一个微信公众号的专属答疑助手。请优先参考下面提供的【知识库参考资料】来解答。\n"
            "如果资料中包含答案，请详细回答。如果资料无关或为空，请使用你的通用知识进行自然回答，不要提及『知识库为空』。\n\n"
            f"【知识库参考资料】开始：\n{reference_context}\n【知识库参考资料】结束。"
        )

    # 3. 组装历史记录
    with pool_lock:
        if user_id not in history_pool:
            history_pool[user_id] = []
            
        history_pool[user_id].append({"role": "user", "content": prompt})
        
        if len(history_pool[user_id]) > MAX_HISTORY:
            history_pool[user_id] = history_pool[user_id][-MAX_HISTORY:]
            
        current_messages = list(history_pool[user_id])
        task_pool[user_id] = {"status": "processing", "result": ""}
    
    system_prompt = {"role": "system", "content": system_role_desc}
    messages_to_send = [system_prompt] + current_messages

    payload = {
        "model": MODEL,
        "messages": messages_to_send,
        "max_tokens": MAX_TOKENS
    }
    
    # 4. 呼叫主模型
    try:
        res = requests.post(API_URL, json=payload, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120)
        data = res.json()
        response_text = data["result"]["response"]
        
        with pool_lock:
            history_pool[user_id].append({"role": "assistant", "content": response_text})
            if len(history_pool[user_id]) > MAX_HISTORY:
                history_pool[user_id] = history_pool[user_id][-MAX_HISTORY:]
                
            task_pool[user_id]["result"] = response_text
            task_pool[user_id]["status"] = "done"
            
    except Exception as e:
        with pool_lock:
            task_pool[user_id]["result"] = f"主模型请求出错: {str(e)}"
            task_pool[user_id]["status"] = "error"
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
# 新增：画图后台逻辑
# ==========================================
def generate_image(user_id, prompt):
    global task_pool
    with pool_lock:
        task_pool[user_id] = {"status": "processing", "result": ""}
    
    ai_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-2-klein-4b"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    
    try:
        # 【关键优化 1】将超时放宽到 120 秒，给 9B 大模型充足的算力时间
        payload = {
            "prompt": (None, prompt) # (文件名/字段名, 内容)
        }
        res = requests.post(ai_url, headers=headers, files=payload, timeout=120)
        
        if res.status_code == 200:
            image_filename = f"{uuid.uuid4().hex}.png"
            image_path = os.path.join("static", image_filename)
            
            # 【关键优化 2】防御性解析：处理 CF 可能返回 base64 JSON 的情况
            content_type = res.headers.get("Content-Type", "")
            if "application/json" in content_type:
                import base64
                data = res.json()
                if "result" in data and "image" in data["result"]:
                    img_bytes = base64.b64decode(data["result"]["image"])
                    with open(image_path, "wb") as f:
                        f.write(img_bytes)
                else:
                    raise Exception("未知的JSON图像返回结构")
            else:
                # 正常的二进制流直接写入
                with open(image_path, "wb") as f:
                    f.write(res.content)
            
            with pool_lock:
                task_pool[user_id]["result"] = image_filename
                task_pool[user_id]["status"] = "done_image" 
        else:
            # 【关键修复】直接把 Cloudflare 真实的 HTTP 状态码和报错内容抛给用户看！
            error_msg = f"HTTP {res.status_code}: {res.text}"
            print(f"日志报错: {error_msg}")
            with pool_lock:
                task_pool[user_id]["result"] = f"出图失败，云端返回: {error_msg}"
                task_pool[user_id]["status"] = "error"
                
    except Exception as e:
        with pool_lock:
            task_pool[user_id]["result"] = f"请求异常或超时: {str(e)}"
            task_pool[user_id]["status"] = "error"

# ==========================================
# 路由修改与新增
# ==========================================
# 【新增 2】画图专用入口
@app.route('/draw', methods=['POST'])
def draw_image():
    if not verify_auth():
        return jsonify({"msg": "error", "error": "Unauthorized"}), 401

    data = request.get_json()
    user_id = data.get("user_id")
    prompt = data.get("prompt")
    
    if not user_id or not prompt:
        return jsonify({"msg": "error", "error": "Missing params"}), 400

    thread = threading.Thread(target=generate_image, args=(user_id, prompt))
    thread.start()
    return jsonify({"msg": "ok"})

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

    use_search = data.get("use_search", False)
    
    if not user_id or not text:
        return jsonify({"msg": "error", "error": "Missing user_id or text"}), 400

    thread = threading.Thread(target=fetch_llm, args=(user_id, text, use_search))
    thread.start()
    
    return jsonify({"msg": "ok"})

# 【修改 3】取件路由增加对图片的特殊分发处理
@app.route('/get_result', methods=['GET'])
def get_result():
    if not verify_auth():
        return jsonify({"status": "error", "data": "Unauthorized"}), 401

    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "data": "Missing user_id"})
        
    with pool_lock:
        if user_id not in task_pool:
            return jsonify({"status": "idle", "data": "当前没有正在处理的问题，请先发送『ai 问题』（ai后面有空格）进行提问。"})
            
        task = task_pool[user_id]
        if task["status"] in ["done", "error"]:
            res = task["result"]
            
            # 【关键修复】微信单条被动回复最多约 600 汉字，为了绝对安全，按 500 字切片
            CHUNK_SIZE = 500
            
            if len(res) > CHUNK_SIZE:
                # 截取前 500 个字符发给用户
                chunk = res[:CHUNK_SIZE]
                # 把剩下的内容重新存回任务池，等待用户下一次取件
                task["result"] = res[CHUNK_SIZE:]
                return jsonify({
                    "status": "done", 
                    "data": chunk + "\n\n...(字数超限，请再次发送『ai 取件』（ai后面有空格）获取剩余内容)"
                })
            else:
                # 如果字数小于 500，一次性发完并彻底销毁任务
                del task_pool[user_id]
                return jsonify({"status": "done", "data": res})

        elif task["status"] == "done_image":
            res = task["result"] # 拿到刚才生成的文件名
            del task_pool[user_id]
            return jsonify({"status": "done_image", "data": res})
            
        elif task["status"] == "processing":
            return jsonify({"status": "processing", "data": "模型还在疯狂输出中，请稍后再次发送『ai 取件』（ai后面有空格）..."})

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
