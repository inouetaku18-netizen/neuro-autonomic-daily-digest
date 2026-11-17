import os
import re
import json
import time
import datetime
import requests
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NCBI_EMAIL = "research-digest@example.com"

# =========================================================================
# 1. 研究トラック定義（PubMed Central Open Access Subset 直接検索）
# =========================================================================
RESEARCH_TRACKS = [
    {
        "category": "point_process",
        "category_label": "点過程・脳心臓連関",
        "query": '("brain-heart" OR "point process" OR "heart rate variability") AND ("electroencephalography" OR "autonomic" OR "interplay") AND "open access"[filter]'
    },
    {
        "category": "rc_koopman",
        "category_label": "RC + Koopman制御",
        "query": '("Koopman operator" OR "reservoir computing") AND ("dynamical systems" OR "control" OR "physiological" OR "neural") AND "open access"[filter]'
    },
    {
        "category": "nonlinear_hrv",
        "category_label": "HRV・非線形時系列解析",
        "query": '("sample entropy" OR "Poincare plot" OR "surrogate data") AND ("heart rate variability" OR "autonomic nervous system") AND "open access"[filter]'
    },
    {
        "category": "gut_brain",
        "category_label": "脳腸相関・迷走神経",
        "query": '("vagus nerve" OR "gut-brain axis") AND ("autonomic" OR "heart rate variability" OR "interoceptive") AND "open access"[filter]'
    }
]

CLINICAL_REVIEW_TOPICS = [
    {
        "domain_label": "認知症の早期バイオマーカーと自律神経・青斑核連関",
        "query": '("locus coeruleus" OR "prodromal Alzheimer" OR "dementia") AND ("autonomic" OR "heart rate variability") AND "open access"[filter]'
    },
    {
        "domain_label": "てんかん・神経疾患における脳心臓連関と迷走神経制御",
        "query": '("epilepsy" OR "vagus nerve stimulation") AND ("heart rate variability" OR "brain-heart") AND "open access"[filter]'
    }
]

# =========================================================================
# 2. 構造化出力用 Pydantic スキーマ
# =========================================================================
class ParagraphPair(BaseModel):
    en: str = Field(description="PMCフルテキストから抽出した実際の英語原文パラグラフ（改変・創作厳禁）")
    ja: str = Field(description="上記英語原文パラグラフの忠実な学術日本語訳")

class PaperSection(BaseModel):
    id: str = Field(description="intro, methods, results, discussion, conclusion のいずれか")
    title_en: str
    title_ja: str
    section_svg: str = Field(description="このセクションの内容を視覚化するクリーンなインラインSVGコード（<svg ...>...</svg>）")
    paragraphs: List[ParagraphPair]

class VocabularyItem(BaseModel):
    term: str = Field(description="英語原文内に実際に登場する学術用語・重要動詞")
    ipa: str = Field(description="国際音声記号（例: /ˌbæroʊˈriːflɛks/）")
    pos: str = Field(description="品詞（noun, verb, adj 等）")
    ja: str = Field(description="学術的な日本語訳")
    nuance: str = Field(description="論文読解・執筆における用法とニュアンス解説")

class PaperSummary(BaseModel):
    problem: str = Field(description="従来研究の課題と本研究の問い（日本語）")
    solution_and_method: str = Field(description="提案手法・実験プロトコル・数理モデルの要点（日本語）")
    formula_block: str = Field(description="論文中の主要な数式または評価指標のLaTeX表記（例: $$...$$）")
    diagram_caption: str = Field(description="概念図のタイトル（英語・日本語併記）")
    diagram_svg: str = Field(description="研究全体のアーキテクチャや生理学的経路を示すインラインSVGコード（<svg ...>...</svg>）")
    key_results: str = Field(description="論文中に実際に記載された定量的結果・統計値（日本語）")
    discussion_and_implications: str = Field(description="著者の考察と生理学的・工学的意義（日本語）")
    relevance_to_my_research: str = Field(description="脳心臓点過程モデル・非線形HRV・RC+Koopman制御への応用ヒント（日本語）")

class DailyPaperSchema(BaseModel):
    title_ja: str = Field(description="論文タイトルの正確な日本語訳")
    impact_factor: float = Field(description="掲載誌の直近インパクトファクター概算値（不明な場合は3.5）")
    tags: List[str] = Field(description="日本語・英語の専門タグ4〜6個")
    summary: PaperSummary
    sections: List[PaperSection]
    vocabulary_list: List[VocabularyItem]

class ReviewPaperEntry(BaseModel):
    pmid: str
    doi: str
    title_en: str
    title_ja: str
    journal: str
    key_finding_ja: str

class ClinicalReviewSchema(BaseModel):
    title_ja: str
    title_en: str
    papers: List[ReviewPaperEntry]
    synthesis: str = Field(description="複数論文を統合した臨床・数理モデリング横断レビュー（日本語）")

# =========================================================================
# 3. PubMed Central (PMC) Open Access フルテキスト実取得関数
# =========================================================================
def sanitize_svg(svg_str: str, min_height: int = 220) -> str:
    if not svg_str or "<svg" not in svg_str:
        return f'<svg viewBox="0 0 600 {min_height}" xmlns="http://www.w3.org/2000/svg"><rect width="600" height="{min_height}" rx="8" fill="#f8fafc" stroke="#cbd5e1"/><text x="300" y="{min_height//2}" text-anchor="middle" fill="#475569" font-size="14">Conceptual Diagram</text></svg>'
    match = re.search(r"<svg[\s\S]*?</svg>", svg_str, re.IGNORECASE)
    return match.group(0) if match else svg_str


def search_pmc_open_access(query: str, max_results: int = 30) -> List[str]:
    """PubMed Central (db=pmc) から Open Access Subset の PMCID リストを取得する"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
        "email": NCBI_EMAIL
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_pmc_article_and_fulltext(numeric_pmcid: str) -> Optional[Tuple[dict, str]]:
    """PMC XMLから実在メタデータ（PMID/PMCID/DOI/Title/Journal/Abstract）と本文フルテキストを一括抽出する"""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pmc", "id": numeric_pmcid, "retmode": "xml", "email": NCBI_EMAIL}
    resp = requests.get(url, params=params, timeout=25)
    if resp.status_code != 200:
        return None

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    article = root.find(".//article")
    if article is None:
        return None

    # ID抽出（PMIDとDOIが両方揃っている査読論文のみ許可）
    pmid = ""
    doi = ""
    pmcid_str = f"PMC{numeric_pmcid}"
    for aid in article.findall(".//article-meta/article-id"):
        id_type = aid.attrib.get("pub-id-type", "")
        if id_type == "pmid" and aid.text:
            pmid = aid.text.strip()
        elif id_type == "doi" and aid.text:
            doi = aid.text.strip()
        elif id_type == "pmc" and aid.text:
            val = aid.text.strip()
            pmcid_str = val if val.startswith("PMC") else f"PMC{val}"

    if not pmid or not doi:
        return None

    # タイトル・ジャーナル名・アブストラクト抽出
    title_el = article.find(".//article-meta/title-group/article-title")
    title_en = " ".join("".join(title_el.itertext()).split()) if title_el is not None else ""
    if not title_en:
        return None

    journal_el = article.find(".//journal-meta/journal-title-group/journal-title")
    if journal_el is None:
        journal_el = article.find(".//journal-meta/journal-title")
    journal = " ".join("".join(journal_el.itertext()).split()) if journal_el is not None else "Peer-Reviewed Open Access Journal"

    abstract_parts = []
    for ab in article.findall(".//article-meta/abstract"):
        txt = " ".join("".join(ab.itertext()).split())
        if txt:
            abstract_parts.append(txt)
    abstract = "\n".join(abstract_parts)

    # 本文（<body>）のセクション・段落抽出
    body = article.find(".//body")
    if body is None:
        return None

    extracted_blocks = []
    total_chars = 0
    for sec in body.findall("./sec"):
        sec_title_el = sec.find("./title")
        sec_title = " ".join("".join(sec_title_el.itertext()).split()) if sec_title_el is not None else "Section"
        paragraphs = []
        for p in sec.findall(".//p"):
            p_text = " ".join("".join(p.itertext()).split())
            if len(p_text) > 40:
                paragraphs.append(p_text)
                total_chars += len(p_text)
        if paragraphs:
            selected_p = "\n\n".join(paragraphs[:4])
            extracted_blocks.append(f"=== SECTION: {sec_title} ===\n{selected_p}")

    if total_chars < 1500 or len(extracted_blocks) < 2:
        return None

    meta = {
        "pmid": pmid,
        "pmcid": pmcid_str,
        "doi": doi,
        "title_en": title_en,
        "journal": journal,
        "abstract": abstract
    }
    full_text = "\n\n".join(extracted_blocks)[:24000]
    return meta, full_text


# =========================================================================
# 4. Gemini API クライアント & 生成ロジック
# =========================================================================
def get_active_client_and_model():
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"api_version": "v1beta"}
    )
    preferred = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.0-flash",
    ]
    available = []
    try:
        for m in client.models.list():
            methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            if not methods or "generateContent" in methods:
                name = m.name.replace("models/", "")
                if "gemini" in name and "embedding" not in name and "imagen" not in name:
                    available.append(name)
    except Exception as e:
        print(f"Model list check skipped: {e}")

    for candidate in preferred:
        if candidate in available:
            print(f"Selected Gemini model: {candidate}")
            return client, candidate

    flash_models = sorted(
        [m for m in available if "flash" in m and "2.0" not in m and "2.5" not in m and "lite" not in m],
        reverse=True
    )
    if flash_models:
        print(f"Selected Gemini model (auto-flash): {flash_models[0]}")
        return client, flash_models[0]

    return client, "gemini-3.8-flash"


def generate_daily_paper_json(meta: dict, full_text: str, track: dict, today_str: str) -> dict:
    client, model_name = get_active_client_and_model()

    prompt = f"""
あなたは生体医工学・神経科学・非線形制御工学の専門研究者兼バイリンガル編集者です。
以下は PubMed Central (PMC) から実際に取得された Open Access 査読済み論文のメタデータおよびフルテキスト原文です。

【厳守ルール（捏造・ハルシネーションの完全禁止）】
1. `sections` 内の各パラグラフの `en` フィールドには、**下記の【PMC Full Text 原文】に実際に書かれている英文パラグラフを一字一句改変・創作せずそのまま引用**してください。元の論文に存在しない英文を作ることは厳禁です。
2. `ja` フィールドには、その `en` パラグラフの忠実で正確な学術日本語訳を記述してください。
3. `sections` は論文の実際のセクション構成に対応させて `intro`, `methods`, `results`, `discussion`, `conclusion` の5つに整理してください（元の論文に独立したConclusionがない場合はDiscussion後半の結論パラグラフを原文のまま割り当ててください）。
4. `summary` の `key_results` には、本文中に実際に記載されている統計値・実験結果のみを記載し、記載のない数値を推測しないでください。
5. `diagram_svg` および各セクションの `section_svg` は、背景淡色（#f8fafc）、青系（#1e3a8a, #2563eb）・ティール系（#0d9488）を基調とした図解SVGを記述してください。
6. `vocabulary_list` には、抽出した `en` テキスト内に実際に登場する専門用語・学術動詞を10〜15個選び解説してください。

【論文メタデータ（実在確認済み）】
PMID: {meta['pmid']}
PMCID: {meta['pmcid']}
DOI: {meta['doi']}
Title: {meta['title_en']}
Journal: {meta['journal']}

【Abstract】
{meta['abstract']}

【PMC Full Text 原文】
{full_text}
"""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DailyPaperSchema,
            temperature=0.1
        )
    )
    parsed = json.loads(response.text)

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
        "pmcid": meta["pmcid"],
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{meta['pmid']}/",
        "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{meta['pmcid']}/",
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
    client, model_name = get_active_client_and_model()
    papers_text = "\n\n".join([
        f"[{i+1}] PMID: {m['pmid']} | PMCID: {m['pmcid']} | DOI: {m['doi']}\nTitle: {m['title_en']}\nJournal: {m['journal']}\nAbstract: {m['abstract']}"
        for i, m in enumerate(papers_meta)
    ])

    prompt = f"""
あなたは臨床神経科学および生体医工学のシニアリサーチャーです。
以下の実際にPubMed Central (PMC) で Open Access 公開されている査読済み論文のみに基づき、「{review_topic['domain_label']}」に関する横断レビューをJSON形式で作成してください。記載のない論文や数値を創作することは厳禁です。

{papers_text}
"""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClinicalReviewSchema,
            temperature=0.1
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


# =========================================================================
# 5. メイン実行フロー
# =========================================================================
def main():
    jst = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(jst)
    today_str = now.strftime("%Y-%m-%d")

    os.makedirs("data/papers", exist_ok=True)
    os.makedirs("data/reviews", exist_ok=True)
    manifest_path = "data/manifest.json"

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"papers": [], "reviews": [], "processed_pmids": []}

    processed_set = set(manifest.get("processed_pmids", []))

    if now.weekday() == 6:
        topic = CLINICAL_REVIEW_TOPICS[now.isocalendar()[1] % len(CLINICAL_REVIEW_TOPICS)]
        pmc_ids = search_pmc_open_access(topic["query"], max_results=20)
        metas = []
        for num_id in pmc_ids:
            res = fetch_pmc_article_and_fulltext(num_id)
            if res:
                metas.append(res[0])
            if len(metas) >= 4:
                break
            time.sleep(0.4)

        if metas:
            review_data = generate_clinical_review_json(metas, topic, today_str)
            out_path = f"data/reviews/{today_str}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(review_data, f, ensure_ascii=False, indent=2)
            manifest["reviews"] = [r for r in manifest["reviews"] if r["id"] != review_data["id"]]
            manifest["reviews"].insert(0, {
                "id": review_data["id"],
                "date": today_str,
                "path": out_path,
                "title_ja": review_data["title_ja"]
            })
    else:
        start_idx = now.timetuple().tm_yday % len(RESEARCH_TRACKS)
        ordered_tracks = RESEARCH_TRACKS[start_idx:] + RESEARCH_TRACKS[:start_idx]

        selected_track = None
        selected_meta = None
        selected_full_text = None

        for track in ordered_tracks:
            print(f"Searching PMC Open Access papers for track: {track['category_label']}")
            pmc_ids = search_pmc_open_access(track["query"], max_results=30)
            print(f"Found {len(pmc_ids)} candidate PMCIDs")

            for num_id in pmc_ids:
                time.sleep(0.4)
                res = fetch_pmc_article_and_fulltext(num_id)
                if not res:
                    continue
                meta, full_text = res
                if meta["pmid"] in processed_set:
                    continue
                selected_track = track
                selected_meta = meta
                selected_full_text = full_text
                print(f"Verified Open Access paper: PMID={meta['pmid']}, PMCID={meta['pmcid']}, DOI={meta['doi']}")
                break

            if selected_meta:
                break

        if not selected_meta or not selected_full_text or not selected_track:
            raise RuntimeError("No new PMC Open Access full-text paper found across all tracks.")

        paper_data = generate_daily_paper_json(selected_meta, selected_full_text, selected_track, today_str)
        out_path = f"data/papers/{today_str}-{selected_meta['pmid']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(paper_data, f, ensure_ascii=False, indent=2)

        manifest["papers"] = [p for p in manifest["papers"] if p["id"] != paper_data["id"]]
        manifest["papers"].insert(0, {
            "id": paper_data["id"],
            "date": today_str,
            "path": out_path,
            "title_ja": paper_data["title_ja"]
        })
        processed_set.add(selected_meta["pmid"])
        manifest["processed_pmids"] = list(processed_set)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Successfully updated manifest.json")


if __name__ == "__main__":
    main()
