'''
This is the main Python file that runs the Flask application and the LLM buffering proxy with session history.
'''
import os
import re
import requests
import random
import uuid
import threading
from flask import Flask, render_template, request, jsonify
from gevent.pywsgi import WSGIServer
from duckduckgo_search import DDGS
from datetime import datetime, timedelta, timezone

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

def clean_markdown_for_wechat(md_text):
    if not md_text:
        return ""
    
    text = md_text

    # 1. 处理 LaTeX 常见符号
    # 把 $\rightarrow$, \rightarrow, -> 等都统一替换成漂亮的 Unicode 箭头
    text = re.sub(r'\$?\\rightarrow\$?', '→', text)
    text = re.sub(r'\$?\\leftarrow\$?', '←', text)
    text = re.sub(r'\$?\\Rightarrow\$?', '⇒', text)
    
    # 2. 转换超链接: [文字](URL) -> <a href="URL">文字</a>
    text = re.sub(r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)', r'<a href="\2">\1</a>', text)
    
    # 3. 转换标题: ### 标题 -> 【标题】
    text = re.sub(r'^#+\s+(.*)$', r'【\1】', text, flags=re.MULTILINE)
    
    # 4. 移除加粗和斜体符号
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # 5. 优化无序列表
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    
    # 6. 清理代码块记号
    text = re.sub(r'```[a-zA-Z]*\n', '\n', text)
    text = re.sub(r'```', '', text)

    # 7. 清理行内单反引号
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # 8. 处理多余的连续换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def find_smart_split_index(text, max_length=600):
    """智能寻找前 max_length 个字符中最合适的标点符号进行断句"""
    if len(text) <= max_length:
        return len(text)
        
    segment = text[:max_length]
    # 定义标点符号的优先级（双换行最高，逗号最低）
    punctuation_marks = ['\n\n', '\n', '。', '！', '？', '.', '!', '?', '；', ';', '，', ',']
    
    for punct in punctuation_marks:
        # 从右向左找，找到这段话里最后一个该类型的标点
        split_index = segment.rfind(punct)
        if split_index != -1:
            # 切割点包含标点符号本身
            return split_index + len(punct)
            
    # 如果极端情况下 600 字里一个标点都没有，就只能强行硬切
    return max_length

def search_knowledge_base(query_text):
    """从 Cloudflare Vectorize 搜索最相关的文本"""
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
# 联网搜索辅助函数
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
# 混合上下文融合架构 (Hybrid Context Fusion)
# ==========================================
def fetch_llm(user_id, prompt):
    global task_pool, history_pool

    # 获取当前准确的北京时间 (UTC+8)
    bj_tz = timezone(timedelta(hours=8))
    current_time = datetime.now(bj_tz).strftime("%Y年%m月%d日 %H:%M %A")

    print(f"🚀启动并行检索流 -> 提问: {prompt}")
    
    # 同时进货：不管三七二十一，两路数据直接全量抓取
    try:
        kb_context = search_knowledge_base(prompt)
    except Exception as e:
        print(f"本地知识库检索异常: {e}")
        kb_context = ""

    try:
        web_context = get_web_search_context(prompt)
    except Exception as e:
        print(f"全网实时搜索异常: {e}")
        web_context = ""

    # 【调试日志】可以在控制台清晰看到两路数据的丰满程度
    print(f"📊[混合检索数据量] 知识库: {len(kb_context if kb_context else '')} 字 | 全网搜索: {len(web_context if web_context else '')} 字")

    # 编写“降维打击”系统提示词，把整合/抛弃的逻辑完全交给大模型的大脑
    system_role_desc = (
        f"你是一个微信公众号的智能助手。【重要提示：当前真实北京时间是 {current_time}】。\n"
        "为了协助你完成最高质量的回答，后台系统已为你同时检索了【本地私有知识库】和【全网实时搜索引擎】。\n\n"
        "请你展现出色的信息检索与整合能力，严格遵循以下融合逻辑：\n"
        "1. 【信息价值评估】：请自行审视下面的【本地知识库参考资料】。如果其内容与用户提问毫无关联、或者仅包含“未找到/抱歉/无匹配”等系统无货填充词，请在心中果断将其评估为【无效干扰信息】，并在后续回答中【完全忽略它】，仅保留并完全基于【联网搜索资料】进行作答。\n"
        "2. 【深度跨域整合】：如果【本地知识库参考资料】包含了与提问强相关的私有知识（如技术参数、私有文档），同时【联网搜索资料】提供了最新的全网时效性进展（如今日动态、最新新闻），请将两份资料【有机融合】。用最新的联网时效补全私有知识，或用私有知识深化网络信息，给出兼具时效性与专业深度的完美回答。\n"
        "3. 【时间锚定守则】：对于任何涉及时间（今天、今年、近一周、最近）的问题，必须结合系统提供的当前真实北京时间进行严格对齐，拒绝对历史死知识的幻觉。\n"
        "4. 【溯源规范】：如果回答中采用了联网搜索的内容，请务必在回答的尾部另起一行附上参考过的网络原文链接。\n\n"
        f"【本地知识库参考资料】开始：\n{kb_context}\n【本地知识库参考资料】结束。\n\n"
        f"【联网搜索资料】开始：\n{web_context}\n【联网搜索资料】结束。"
    )

    # 组装历史记录
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
    
    # 向Cloudflare发起最终请求
    try:
        res = requests.post(API_URL, json=payload, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120)
        if res.status_code == 200:
            data = res.json()
            response_text = data["choices"][0]["message"]["content"]
            
            with pool_lock:
                history_pool[user_id].append({"role": "assistant", "content": response_text})
                if len(history_pool[user_id]) > MAX_HISTORY:
                    history_pool[user_id] = history_pool[user_id][-MAX_HISTORY:]

                # 给用户展示用的结果（存入 task_pool 等待微信服务器拉取）
                # 只有在这里，才调用降维清洗函数
                cleaned_text_for_user = clean_markdown_for_wechat(response_text)
                
                task_pool[user_id]["result"] = cleaned_text_for_user
                task_pool[user_id]["status"] = "done"
        else:
            raise Exception(f"HTTP {res.status_code}: {res.text}")
    except Exception as e:
        print(f"请求失败: {e}")
        with pool_lock:
            task_pool[user_id]["result"] = f"处理失败，错误原因: {str(e)}"
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
# 画图后台逻辑
# ==========================================
def generate_image(user_id, prompt):
    global task_pool
    with pool_lock:
        task_pool[user_id] = {"status": "processing", "result": ""}
    
    ai_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-2-klein-4b"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    
    try:
        # 将超时放宽到 120 秒，给大模型充足的算力时间
        payload = {
            "prompt": (None, prompt) # (文件名/字段名, 内容)
        }
        res = requests.post(ai_url, headers=headers, files=payload, timeout=120)
        
        if res.status_code == 200:
            image_filename = f"{uuid.uuid4().hex}.png"
            image_path = os.path.join("static", image_filename)
            
            # 防御性解析：处理 CF 可能返回 base64 JSON 的情况
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
            # 直接把 Cloudflare 真实的 HTTP 状态码和报错内容抛给用户看！
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
# 画图专用入口
# ==========================================
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
    
    if not user_id or not text:
        return jsonify({"msg": "error", "error": "Missing user_id or text"}), 400

    thread = threading.Thread(target=fetch_llm, args=(user_id, text))
    thread.start()
    
    return jsonify({"msg": "ok"})

# 取件路由增加对图片的特殊分发处理
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
            
            CHUNK_SIZE = 600
            
            if len(res) > CHUNK_SIZE:
                # 调用智能断句函数
                split_index = find_smart_split_index(res, CHUNK_SIZE)
                # 截取到标点符号处，发给用户
                chunk = res[:split_index].strip()
                # 把剩下的内容重新存回任务池，等待用户下一次取件
                task["result"] = res[split_index:].strip()
                return jsonify({
                    "status": "done", 
                    "data": chunk + "\n\n...(字数超限，请再次发送『ai 取件』（ai后面有空格）获取剩余内容)"
                })
            else:
                # 如果字数小于 600，一次性发完并彻底销毁任务
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
