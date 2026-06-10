import os
import uuid
import re
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser
import pandas as pd
import requests
import google.generativeai as genai
from pydantic import BaseModel, Field
# 引入 Supabase 套件
from supabase import create_client, Client

# 引入 Google API 的頻率限制異常類型
from google.api_core.exceptions import ResourceExhausted

# ==========================================
# 1. 基礎配置與時區/時間窗口校正
# ==========================================
TZ_TW = timezone(timedelta(hours=8))
NOW_TW = datetime.now(TZ_TW)
TODAY_STR = NOW_TW.strftime("%Y-%m-%d")
YESTERDAY_STR = (NOW_TW - timedelta(days=1)).strftime("%Y-%m-%d")

# 定義 48 小時滑動窗口，避免任何因時差或排程造成的情報盲區
TIME_WINDOW = timedelta(hours=48)

# ------ AI 頻率限制配置 ------
MAX_RPM = 10  # 如果你是付費帳戶，可以調高（例如 1000）；免費版通常是 15
# 計算每次請求之間理論上應間隔的秒數，留一點緩衝（例如 15 RPM = 每 4 秒多發一次）
BASE_DELAY = (60.0 / MAX_RPM) + 0.2 

# 驗證環境變數
REQUIRED_ENV = ["GEMINI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
for env in REQUIRED_ENV:
    if not os.environ.get(env):
        raise ValueError(f"環境變數中缺少 {env}，請先設定。")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# 初始化 Supabase 用戶端
supabase_url = os.environ["SUPABASE_URL"]
supabase_key = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# ==========================================
# 2. 18 大權威情資來源註冊表 (CTI Source Registry)
# ==========================================
CTI_REGISTRY = {
    # --- 1. 權威資安新聞與專業媒體 ---
    "BleepingComputer": {"format": "RSS", "url": "https://www.bleepingcomputer.com/feed/"},
    "SecurityWeek": {"format": "RSS", "url": "https://feeds.feedburner.com/securityweek"},
    "DarkReading": {"format": "RSS", "url": "https://www.darkreading.com/rss.xml"},
    "CyberScoop": {"format": "RSS", "url": "https://cyberscoop.com/feed/"},
    "CybersecurityDive": {"format": "RSS", "url": "https://www.cybersecuritydive.com/feed/"},

    # --- 2. 漏洞公告與原廠安全通報 ---
    "MSRC_Blog": {"format": "RSS", "url": "https://msrc.microsoft.com/blog/feed"},
    "Google_Chrome_Releases": {"format": "RSS", "url": "https://chromereleases.googleblog.com/feeds/posts/default"},
    "Cisco_Advisories": {"format": "RSS", "url": "https://tools.cisco.com/security/center/rss.x?i=44"},
    "PaloAlto_Advisories": {"format": "RSS", "url": "https://security.paloaltonetworks.com/rss.xml"},

    # --- 3. 國家級資安應變中心與執法機構 ---
    "Fortinet_PSIRT": {"format": "RSS", "url": "https://www.fortinet.com/content/fortinet-blog/us/en/rss-feeds/psirt.rss"},
    "CISA_KEV": {"format": "CISA_API", "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"},
    "Singapore_CSA": {"format": "RSS", "url": "https://www.csa.gov/rss/alerts-and-advisories"},
    "Taiwan_TWCERT": {"format": "RSS", "url": "https://www.twcert.org.tw/tw/lp-132-1.xml"},
    "FBI_IC3": {"format": "RSS", "url": "https://www.ic3.gov/Home/RssAlerts"},

    # --- 4. 資安實驗室與威脅情報部落格 ---
    "Cisco_Talos": {"format": "RSS", "url": "https://blog.talosintelligence.com/rss/"},
    "Mandiant_Blog": {"format": "RSS", "url": "https://www.mandiant.com/resources/blog/rss.xml"},
    "TrendMicro_Research": {"format": "RSS", "url": "https://feeds.trendmicro.com/TrendMicroSecurityNews"},
    "Malwarebytes_Labs": {"format": "RSS", "url": "https://www.malwarebytes.com/blog/feed"},
    "SentinelOne_Blog": {"format": "RSS", "url": "https://www.sentinelone.com/blog/feed/"},
    
    # --- 5. 開源代碼庫與漏洞概念驗證 (範例 API) ---
    "GitHub_Exploit_Search": {"format": "GITHUB_API", "url": f"https://api.github.com/search/repositories?q=created:%3E={YESTERDAY_STR}+topic:exploit&sort=stars"}
}

# ==========================================
# Pydantic 結構化資料模型
# ==========================================
class SecurityEnrichment(BaseModel):
    threat_type: str = Field(description="威脅類型。例如：DDoS、漏洞、資料外洩、漏洞修補等")
    severity: str = Field(description="嚴重程度。必須嚴格填入 'High' 或 'Medium' 或 'Normal'")
    location: str = Field(description="事發地點名稱。例如：台灣, 台北、美國、全球。若無特定則填全球")
    lat: float = Field(description="該地點的緯度。未知或全球則填 0.0000")
    lng: float = Field(description="該地點的經度。未知或全球則填 0.0000")
    Summary: str = Field(description="將原文精煉並翻譯為繁體中文的新聞內容摘要，不超過150字")
    Suggestion: str = Field(description="站在資安專家角度，針對該事件給出具體、可執行的繁體中文處置或緩解作法建議")

# ==========================================
# AI 核心語意增強引擎 (LLM Engine) - 已整合重試與防爆機制
# ==========================================
def ai_enrichment_engine(title: str, raw_summary: str, max_retries: int = 5) -> SecurityEnrichment:
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    
    prompt = f"""
    你是一個頂尖的威脅情報分析師 (Cyber Threat Intelligence Analyst)。請精確剖析以下資安情資，並輸出結構化 JSON。
    
    【標題】：{title}
    【內容描述】：{raw_summary}
    
    規範：
    1. Summary 與 Suggestion 必須為繁體中文（台灣），用詞需符合專業 CISSP 術語（如：在野利用、社交工程、憑證竊取）。
    2. 依據內容精準賦予 severity (High/Medium/Normal)。
    3. 準確識別地理座標。若提及特定國家受災，給出該國首都經緯度；若屬全球通用軟體漏洞，lat/lng 填 0.0000，location 填 '全球'。
    """
    
    for attempt in range(max_retries):
        try:
            # 正常狀況下的配速防線
            if attempt == 0:
                time.sleep(BASE_DELAY)
                
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", response_schema=SecurityEnrichment, temperature=0.1
                )
            )
            return SecurityEnrichment.model_validate_json(response.text)
            
        except ResourceExhausted:
            # 專門捕捉 429 錯誤，進行指數退避 (Exponential Backoff)
            # 第一次重試等 BASE*2，第二次等 BASE*4，依此類推
            wait_time = BASE_DELAY * (2 ** attempt)
            print(f"  [RPM Limit] 觸發 API 頻率上限！將在 {wait_time:.1f} 秒後進行第 {attempt + 1}/{max_retries} 次重試...")
            time.sleep(wait_time)
            
        except Exception as e:
            # 捕捉其他與 RPM 無關的錯誤（如模型解析失敗、格式不符等），直接跳出不浪費時間重試
            print(f"  [AI Error] 處理失敗 (非頻率問題): {e}")
            return None
            
    print(f"  [AI Error] 已達最大重試次數 ({max_retries})，放棄此筆資料。")
    return None

# ==========================================
# 解耦的異質資料解析器 (Parsers)
# ==========================================
def fetch_and_parse(name, config):
    local_results = []
    fmt = config["format"]
    url = config["url"]
    try:
        if fmt == "RSS":
            feed = feedparser.parse(url)
            for entry in feed.entries:
                pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).astimezone(TZ_TW) if hasattr(entry, "published_parsed") and entry.published_parsed else NOW_TW
                if NOW_TW - pub_time <= TIME_WINDOW:
                    summary_raw = getattr(entry, "summary", getattr(entry, "description", ""))
                    clean_content = re.sub('<[^<]+?>', '', summary_raw)[:500]
                    local_results.append({
                        "title": entry.title, 
                        "url": entry.link, 
                        "raw_content": clean_content, 
                        "created_at": pub_time.isoformat()
                    })
                    
        elif fmt == "CISA_API":
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                for v in res.json().get("vulnerabilities", []):
                    date_added = v.get("dateAdded", "")
                    if date_added in [TODAY_STR, YESTERDAY_STR]:
                        local_results.append({
                            "title": f"CISA KEV 警訊: {v.get('cveID')} - {v.get('vulnerabilityName')}",
                            "url": f"https://nvd.nist.gov/vuln/detail/{v.get('cveID')}",
                            "raw_content": f"{v.get('shortDescription')} Action: {v.get('requiredAction')}",
                            "created_at": f"{date_added}T08:00:00+08:00"
                        })
                        
        elif fmt == "GITHUB_API":
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if res.status_code == 200:
                for item in res.json().get("items", [])[:5]:
                    local_results.append({
                        "title": f"GitHub 全新漏洞 PoC: {item.get('full_name')}",
                        "url": item.get("html_url"),
                        "raw_content": item.get("description") or "打上了 exploit 標籤的項目。",
                        "created_at": item.get("created_at", NOW_TW.isoformat())
                    })
    except Exception as e:
        print(f"[Fetch Error] 管道 {name} 發生異常: {e}")
    return local_results

# ==========================================
# 3. Supabase 資料庫上傳邏輯 (使用 Upsert)
# ==========================================
def upload_to_supabase(data_list):
    print(f"\n[Supabase] 開始將 {len(data_list)} 筆資料同步至 Supabase...")
    if not data_list:
        return
    try:
        response = supabase.table("News").upsert(data_list, on_conflict="url,title").execute()
        print(f"[Supabase SUCCESS] 成功寫入/更新資料庫！")
    except Exception as e:
        print(f"[Supabase ERROR] 寫入資料庫失敗: {e}")

# ==========================================
# 4. 主控制管線
# ==========================================
def main():
    print(f"=== 啟動 2026 全通路 CTI 融合與 Supabase 同步系統 ===")
    print(f"目前時間 (TW): {NOW_TW.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"時間比對窗口: 過去 48 小時內 ({YESTERDAY_STR} 至 {TODAY_STR})")
    print(f"==================================================")
    
    raw_reports = []
    seen_urls = set()
    final_rows = []

    # 步驟一：多執行緒平行抓取 18 個來源
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_source = {executor.submit(fetch_and_parse, name, cfg): name for name, cfg in CTI_REGISTRY.items()}
        for future in as_completed(future_to_source):
            source_name = future_to_source[future]
            fetched_data = future.result()
            if fetched_data:
                print(f" -> 管道 [{source_name}] 命中 {len(fetched_data)} 筆符合時間窗口之情報！")
                raw_reports.extend(fetched_data)

    # 步驟二：全域去重與 AI 語意強化 (此處保持單執行緒循序呼叫，搭配動態配速)
    print("\n[AI Parsing] 開始進行 AI 結構化強化...")
    for report in raw_reports:
        unique_meta = (report["url"], report["title"])
        if unique_meta in seen_urls:
            continue
        seen_urls.add(unique_meta)
        
        ai_enriched = ai_enrichment_engine(report["title"], report["raw_content"])
        if ai_enriched:
            final_rows.append({
                "title": report["title"],
                "url": report["url"],
                "threat_type": ai_enriched.threat_type,
                "severity": ai_enriched.severity,
                "location": ai_enriched.location,
                "lat": ai_enriched.lat,
                "lng": ai_enriched.lng,
                "created_at": report["created_at"],
                "summary": ai_enriched.Summary,
                "suggestion": ai_enriched.Suggestion
            })

    # 步驟三：匯出合規 CSV 檔與同步
    if final_rows:
        upload_to_supabase(final_rows)
        df = pd.DataFrame(final_rows)
        df.to_csv(f"cti_backup_{TODAY_STR}.csv", index=False, encoding="utf-8-sig")
        print(f"\n[SUCCESS] 全部流程執行完畢，共處理解析 {len(final_rows)} 筆不重複情資。")
    else:
        print(f"\n[INFO] 本次執行無新情資需要上傳。")

if __name__ == "__main__":
    main()
