import os
import re
import json
import time
import datetime
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# 1. 設定・検索クエリ（研究領域ローテーション）
# ==========================================
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "researcher@example.com")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

DATA_DIR = "data"
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
REVIEWS_DIR = os.path.join(DATA_DIR, "reviews")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

# 曜日ごとに検索領域をローテーション（0:月〜5:土は個別論文、6:日は臨床複数論文レビュー）
SEARCH_TRACKS = [
    {
        "category": "koopman",
        "category_label": "RC + Koopman制御",
        "query": '("Koopman operator" OR "reservoir computing" OR "echo state network") AND ("physiological" OR "heart rate" OR "EEG" OR "autonomic" OR "nonlinear control" OR "model predictive control") AND ("Journal Article"[Publication Type]) NOT ("Preprint"[Publication Type])'
    },
    {
        "category": "point_process",
        "category_label": "点過程・脳心臓連関",
        "query": '("brain-heart interplay" OR "central autonomic network" OR ("point process" AND "heart rate variability") OR "heartbeat evoked potential") AND ("Journal Article"[Publication Type]) NOT ("Preprint"[Publication Type])'
    },
    {
        "category": "nonlinear_hrv",
        "category_label": "HRV・非線形時系列解析",
        "query": '("heart rate variability" AND ("sample entropy" OR "Poincare plot" OR "surrogate data" OR "Laguerre" OR "nonlinear dynamics" OR "orthostatic")) AND ("Journal Article"[Publication Type]) NOT ("Preprint"[Publication Type])'
    },
    {
        "category": "gut_brain",
        "category_label": "脳腸相関・迷走神経刺激",
        "query": '("vagus nerve" AND ("gut-brain" OR "GABA" OR "taVNS" OR "afferent" OR "electroencephalography")) AND ("Journal Article"[Publication Type]) NOT ("Preprint"[Publication Type])'
    }
]

CLINICAL_REVIEW_QUERIES = [
    {
        "domain_label": "認知症メカニズム・超早期診断",
        "query": '("dementia" OR "Alzheimer" OR "mild cognitive impairment") AND ("heart rate variability" OR "autonomic" OR "locus coeruleus" OR "EEG entropy") AND ("early diagnosis" OR "biomarker") AND ("Journal Article"[Publication Type])'
    },
    {
        "domain_label": "神経心臓連関・精神疾患・てんかん臨床応用",
        "query": '("brain-heart" OR "vagus nerve stimulation" OR "neuro-cardiac") AND ("depression" OR "epilepsy" OR "SUDEP" OR "stroke") AND ("Journal Article"[Publication Type])'
    }
]

# ==========================================
# 2. Pydanticスキーマ定義（index.htmlと完全同期）
# ==========================================
class ParagraphPair(BaseModel):
    ja: str = Field(description="自然で学術的に正確な日本語訳")
    en: str = Field(description="対応する英語原文（1段落または意味のまとまり）")

class WritingExpression(BaseModel):
    pattern: str = Field(description="英語論文執筆に使える汎用構文テンプレート")
    meaning: str = Field(description="日本語での意味・使われる文脈")
    example: str = Field(description="本論文の文脈に即した英語例文")

class FullTextSection(BaseModel):
    id: str = Field(description="intro, methods, results, discussion, conclusion のいずれか")
    title_en: str = Field(description="例: 1. Introduction")
    title_ja: str = Field(description="例: 1. イントロダクション（研究背景と課題）")
    paragraphs: List[ParagraphPair]
    takeaways: List[str] = Field(description="このセクションの重要ポイント3〜4点（日本語）")
    section_svg_caption: str = Field(description="セクション末の図解キャプション")
    section_svg: str = Field(description="このセクションの要点を図解する有効なインラインSVGコード（viewBox='0 0 680 210'）")
    writing_expressions: List[WritingExpression]

class VocabEntry(BaseModel):
    term: str = Field(description="英語本文中に登場する重要な学術用語・熟語（小文字基準）")
    ipa: str = Field(description="発音記号（例: /ˈveɪɡəl wɪðˈdrɔːəl/）")
    pos: str = Field(description="品詞（例: 名詞句, 動詞, 形容詞）")
    ja: str = Field(description="日本語訳")
    nuance: str = Field(description="論文読解・執筆上のニュアンス解説")

class PaperSummary(BaseModel):
    problem: str = Field(description="【扱っている問題点】従来の解析や制御が抱えていた具体的課題")
    solution_and_method: str = Field(description="【解決策と具体的手法】数式やアルゴリズムを用いた手法の詳細解説")
    formula_block: str = Field(description="核心となる数式（LaTeX形式 $$ ... $$ で記述）")
    diagram_caption: str = Field(description="手法概念図のキャプション")
    diagram_svg: str = Field(description="手法の信号処理・状態空間・神経回路フローを示す詳細なインラインSVG（viewBox='0 0 680 240'）")
    key_results: str = Field(description="【主要な結果】定量指標、統計的有意差、過渡応答への追従結果")
    discussion_and_implications: str = Field(description="【重要な考察】生理学的解釈、限界点、今後の展望")
    relevance_to_my_research: str = Field(description="【自分の研究への応用ヒント】点過程モデル（mEEG→RR）、3次元Poincare plot（IIS）、能動的起立試験、RC+Koopman線形制御への具体的な応用アイデア")

class DailyPaperSchema(BaseModel):
    title_ja: str
    impact_factor: float = Field(description="当該ジャーナルの最新Impact Factor（不明な場合は同分野の標準値または推定値を小数1桁で記載）")
    tags: List[str]
    summary: PaperSummary
    sections: List[FullTextSection] = Field(description="必ず intro, methods, results, discussion, conclusion の5セクションを順に出力")
    vocabulary_list: List[VocabEntry] = Field(description="英語本文中に含まれる重要な学術表現10〜15語")

class ReviewSubPaper(BaseModel):
    title: str
    journal: str
    year: int
    impact_factor: float
    doi: str
    finding: str = Field(description="この論文が示した核心的知見の日本語要約")

class ReviewSynthesis(BaseModel):
    current_state: str = Field(description="世界の研究の現在地・確立されつつあるコンセンサス")
    key_mechanisms: str = Field(description="解明された主要な生理学的・数理的メカニズム")
    open_challenges: str = Field(description="現在残されている未解決の技術的・臨床的課題")
    bridge_to_our_research: str = Field(description="秒単位点過程モデル・IIS・RC+Koopman制御がどのようにこの領域を突破できるか")

class ClinicalReviewSchema(BaseModel):
    title_ja: str
    title_en: str
    papers: List[ReviewSubPaper]
    synthesis: ReviewSynthesis

# ==========================================
# 3. PubMed / PMC データ取得関数
# ==========================================
def load_manifest() -> dict:
    os.makedirs(PAPERS_DIR, exist_ok=True)
    os.makedirs(REVIEWS_DIR, exist_ok=True)
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"papers": [], "reviews": [], "processed_pmids": []}

def save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def search_pubmed(query: str, retmax: int = 15) -> List[str]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
        "email": NCBI_EMAIL
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])

def fetch_pubmed_metadata(pmid: str) -> dict:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "email": NCBI_EMAIL}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    title = "".join(root.find(".//ArticleTitle").itertext()).strip()
    journal = "".join(root.find(".//Journal/Title").itertext()).strip()
    
    abstract_parts = []
    for ab in root.findall(".//Abstract/AbstractText"):
        label = ab.attrib.get("Label", "")
        text = "".join(ab.itertext()).strip()
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts)

    doi = ""
    pmcid = ""
    for aid in root.findall(".//ArticleIdList/ArticleId"):
        id_type = aid.attrib.get("IdType")
        if id_type == "doi":
            doi = aid.text.strip()
        elif id_type == "pmc":
            pmcid = aid.text.strip()

    return {
        "pmid": pmid,
        "title_en": title,
        "journal": journal,
        "doi": doi,
        "pmcid": pmcid,
        "abstract": abstract
    }

def fetch_pmc_fulltext(pmcid: str) -> str:
    """PubMed CentralからOpen Access全文XMLを取得してセクション別に整形"""
    if not pmcid:
        return ""
    numeric_id = pmcid.replace("PMC", "")
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pmc", "id": numeric_id, "retmode": "xml", "email": NCBI_EMAIL}
    try:
        resp = requests.get(url, params=params, timeout=45)
        if resp.status_code != 200:
            return ""
        root = ET.fromstring(resp.text)
        body = root.find(".//body")
        if body is None:
            return ""
        sections_text = []
        for sec in body.findall("./sec"):
            sec_title_el = sec.find("./title")
            sec_title = "".join(sec_title_el.itertext()).strip() if sec_title_el is not None else "Section"
            paras = ["".join(p.itertext()).strip() for p in sec.findall(".//p")]
            if paras:
                sections_text.append(f"### {sec_title}\n" + "\n\n".join(paras))
        full_text = "\n\n".join(sections_text)
        # トークン上限保護（最大約45,000文字）
        return full_text[:45000]
    except Exception as e:
        print(f"PMC fulltext fetch skipped ({pmcid}): {e}")
        return ""

# ==========================================
# 4. Gemini APIによる構造化翻訳・SVG図解生成
# ==========================================
def sanitize_svg(svg_str: str, height: int = 230) -> str:
    match = re.search(r"<svg[\s\S]*?</svg>", svg_str, re.IGNORECASE)
    if match:
        return match.group(0)
    return f'<svg viewBox="0 0 680 {height}" class="w-full h-auto"><rect width="680" height="{height}" rx="8" fill="#f8fafc"/><text x="340" y="{height//2}" font-size="13" fill="#475569" text-anchor="middle">Conceptual Architecture Diagram</text></svg>'

def generate_daily_paper_json(meta: dict, full_text: str, track: dict, today_str: str) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    source_material = f"【Full Text (Open Access)】\n{full_text}" if full_text else f"【Abstract】\n{meta['abstract']}"
    
    prompt = f"""
あなたは生体医工学・神経科学・非線形制御工学の専門研究者兼バイリンガル編集者です。
以下の査読済み英語論文を解析し、指定されたJSONスキーマに従って日本語翻訳・構造化要約・数式解説・SVG概念図・全5セクション対訳・学術英単語辞書を生成してください。

【対象者の研究背景（応用ヒント作成の基準）】
- 脳波サンプルエントロピー（mEEG）とR-R間隔の多変量非均質点過程モデルによる秒単位（Δt=1〜5秒）の中枢-自律神経評価
- 経口GABA摂取や脳腸相関（迷走神経求心路）の超高速応答評価
- 3次元Poincaré plot上の局所エントロピー（IIS）およびサロゲートデータ検定による非線形HRV解析
- リザバーコンピューティング（RC）＋クープマン作用素（Koopman operator）による能動的起立試験の固有値解析および非線形生体システムの線形モデル予測制御（MPC）

【要件】
1. `sections` は必ず `intro`, `methods`, `results`, `discussion`, `conclusion` の5つを含めてください。全文テキストがある場合は省略せず段落ペア（日英）に分割して収録し、アブストラクトのみの場合も各セクションの論理構成に沿って詳細に日英対訳を構成してください。
2. `diagram_svg` および各セクションの `section_svg` は、背景白または淡色（#f8fafc）、青系（#1e3a8a, #2563eb）・ティール系（#0d9488）を基調とした見やすい図解SVG（矢印、ブロック図、数式ラベル、波形イメージ等）を直接記述してください。
3. `vocabulary_list` には、生成した `en` テキスト内に実際に登場する単語・フレーズを10〜15個選び、発音記号・日本語訳・論文での使い方を解説してください。

【論文メタデータ】
Title: {meta['title_en']}
Journal: {meta['journal']}
DOI: {meta['doi']}

{source_material}
"""
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DailyPaperSchema,
            temperature=0.2
        )
    )
    parsed = json.loads(response.text)

    # SVGサニタイズと語彙辞書のマップ変換
    parsed["summary"]["diagram_svg"] = sanitize_svg(parsed["summary"]["diagram_svg"], 240)
    for sec in parsed["sections"]:
        sec["section_svg"] = sanitize_svg(sec["section_svg"], 210)

    vocab_dict = {}
    for v in parsed.pop("vocabulary_list", []):
        vocab_dict[v["term"].lower()] = {
            "ipa": v["ipa"],
            "pos": v["pos"],
            "ja": v["ja"],
            "nuance": v["nuance"]
        }

    return {
        "id": f"paper-{today_str}-{meta['pmid']}",
        "date": today_str,
        "pmid": meta["pmid"],
        "category": track["category"],
        "category_label": track["category_label"],
        "journal": meta["journal"],
        "impact_factor": parsed["impact_factor"],
        "doi": meta["doi"],
        "title_en": meta["title_en"],
        "title_ja": parsed["title_ja"],
        "tags": parsed["tags"],
        "summary": parsed["summary"],
        "full_text": {
            "sections": parsed["sections"]
        },
        "vocabulary": vocab_dict
    }

def generate_clinical_review_json(papers_meta: List[dict], review_topic: dict, today_str: str) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)
    papers_text = "\n\n".join([
        f"[{i+1}] Title: {m['title_en']}\nJournal: {m['journal']}\nDOI: {m['doi']}\nAbstract: {m['abstract']}"
        for i, m in enumerate(papers_meta)
    ])

    prompt = f"""
あなたは臨床神経科学および生体医工学のシニアリサーチャーです。
以下の複数本の査読済み論文を横断的に統合し、「{review_topic['domain_label']}」に関する臨床最前線レビュー（世界の研究の現状・メカニズム・未解決課題・点過程/RC+Koopman研究との架け橋）をJSON形式で作成してください。

{papers_text}
"""
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClinicalReviewSchema,
            temperature=0.2
        )
    )
    parsed = json.loads(response.text)
    return {
        "id": f"rev-{today_str}",
        "date": today_str,
        "domain_label": review_topic["domain_label"],
        "title_ja": parsed["title_ja"],
        "title_en": parsed["title_en"],
        "paper_count": len(parsed["papers"]),
        "papers": parsed["papers"],
        "synthesis": parsed["synthesis"]
    }

# ==========================================
# 5. メイン実行フロー
# ==========================================
def main():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY が設定されていません。")

    jst = datetime.timezone(datetime.timedelta(hours=9))
    now_jst = datetime.datetime.now(jst)
    today_str = now_jst.strftime("%Y-%m-%d")
    weekday = now_jst.weekday()  # 0:月 〜 6:日

    manifest = load_manifest()
    processed = set(manifest.get("processed_pmids", []))

    # 日曜日は「臨床複数論文統合レビュー」を生成、月〜土は「毎日の個別論文」を生成
    if weekday == 6:
        topic = CLINICAL_REVIEW_QUERIES[(now_jst.isocalendar()[1]) % len(CLINICAL_REVIEW_QUERIES)]
        pmids = search_pubmed(topic["query"], retmax=12)
        target_pmids = [p for p in pmids if p not in processed][:4]
        if len(target_pmids) < 2:
            target_pmids = pmids[:4]

        metas = []
        for p in target_pmids:
            metas.append(fetch_pubmed_metadata(p))
            processed.add(p)
            time.sleep(0.4)

        review_data = generate_clinical_review_json(metas, topic, today_str)
        rel_path = f"data/reviews/{today_str}.json"
        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)

        manifest["reviews"].insert(0, {
            "id": review_data["id"],
            "date": today_str,
            "path": rel_path,
            "title_ja": review_data["title_ja"]
        })
        print(f"Generated clinical review: {rel_path}")
    else:
        track = SEARCH_TRACKS[weekday % len(SEARCH_TRACKS)]
        pmids = search_pubmed(track["query"], retmax=20)
        selected_pmid = next((p for p in pmids if p not in processed), pmids[0])

        meta = fetch_pubmed_metadata(selected_pmid)
        time.sleep(0.4)
        full_text = fetch_pmc_fulltext(meta["pmcid"])

        paper_data = generate_daily_paper_json(meta, full_text, track, today_str)
        rel_path = f"data/papers/{today_str}.json"
        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump(paper_data, f, ensure_ascii=False, indent=2)

        processed.add(selected_pmid)
        manifest["papers"].insert(0, {
            "id": paper_data["id"],
            "date": today_str,
            "path": rel_path,
            "title_ja": paper_data["title_ja"]
        })
        print(f"Generated daily paper: {rel_path} (FullText={'Yes' if full_text else 'Abstract-only'})")

    manifest["processed_pmids"] = list(processed)
    save_manifest(manifest)

if __name__ == "__main__":
    main()
