import os
import re
import json
import time
import datetime
import requests
import xml.etree.ElementTree as ET
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NCBI_EMAIL = "research-digest@example.com"

# =========================================================================
# 1. 毎日取得する5分野の研究トラック（すべて Free Full Text / OA 限定）
# =========================================================================
DAILY_RESEARCH_TRACKS = [
    {
        "category": "brain",
        "category_label": "① 脳・神経ダイナミクス",
        "query": '("electroencephalography"[Title/Abstract] OR "neural dynamics"[Title/Abstract] OR "cortical oscillations"[Title/Abstract]) AND ("sample entropy"[Title/Abstract] OR "complexity"[Title/Abstract] OR "nonlinear"[Title/Abstract]) AND "free full text"[filter]'
    },
    {
        "category": "heart",
        "category_label": "② 心臓・非線形HRV",
        "query": '("heart rate variability"[Title/Abstract] OR "baroreflex"[Title/Abstract]) AND ("sample entropy"[Title/Abstract] OR "Poincare plot"[Title/Abstract] OR "surrogate"[Title/Abstract] OR "nonlinear"[Title/Abstract]) AND "free full text"[filter]'
    },
    {
        "category": "brain_heart",
        "category_label": "③ 脳-心臓連関・迷走神経",
        "query": '("brain-heart"[Title/Abstract] OR "central autonomic network"[Title/Abstract] OR "vagus nerve"[Title/Abstract] OR "gut-brain axis"[Title/Abstract]) AND ("heart rate variability"[Title/Abstract] OR "electroencephalography"[Title/Abstract] OR "interoceptive"[Title/Abstract]) AND "free full text"[filter]'
    },
    {
        "category": "ai_control",
        "category_label": "④ AI・数理神経制御",
        "query": '("Koopman operator"[Title/Abstract] OR "reservoir computing"[Title/Abstract] OR "closed-loop control"[Title/Abstract] OR "state-space"[Title/Abstract]) AND ("neural"[Title/Abstract] OR "autonomic"[Title/Abstract] OR "physiological"[Title/Abstract]) AND "free full text"[filter]'
    },
    {
        "category": "clinical",
        "category_label": "⑤ 臨床応用・バイオマーカー",
        "query": '("prodromal Alzheimer"[Title/Abstract] OR "dementia"[Title/Abstract] OR "epilepsy"[Title/Abstract] OR "vagus nerve stimulation"[Title/Abstract] OR "locus coeruleus"[Title/Abstract]) AND ("autonomic"[Title/Abstract] OR "heart rate variability"[Title/Abstract]) AND "free full text"[filter]'
    }
]

# =========================================================================
# 2. 構造化出力用 Pydantic スキーマ（アブストラクト事実ベース限定）
# =========================================================================
class VocabularyItem(BaseModel):
    term: str = Field(description="英語アブストラクト内に実際に登場する学術用語・重要動詞")
    ipa: str = Field(description="国際音声記号（例: /ˌbæroʊˈriːflɛks/）")
    pos: str = Field(description="品詞（noun, verb, adj 等）")
    ja: str = Field(description="学術的な日本語訳")
    nuance: str = Field(description="論文読解・執筆における用法とニュアンス解説")

class AbstractDigestSchema(BaseModel):
    title_ja: str = Field(description="英語論文タイトルの正確な日本語訳")
    abstract_ja: str = Field(description="英語アブストラクト原文の一文一文に対応した忠実な日本語訳（推測・補完禁止）")
    impact_factor: float = Field(description="掲載誌の直近インパクトファクター概算値（不明な場合は3.5）")
    tags: List[str] = Field(description="専門タグ4〜6個")
    problem: str = Field(description="アブストラクトに記載された研究背景・課題（記載のない推測は禁止）")
    solution_and_method: str = Field(description="アブストラクトに記載された手法・対象・解析方法（記載のない推測は禁止）")
    formula_block: str = Field(description="この分野・手法に関連する基本数式または評価指標のLaTeX表記（例: $$...$$）")
    diagram_caption: str = Field(description="概念図のタイトル（英語・日本語併記）")
    diagram_svg: str = Field(description="アブストラクトで述べられている因果関係・処理フローを示すクリーンなインラインSVG（<svg ...>...</svg>）")
    key_results: str = Field(description="アブストラクトに明記されている結果・数値のみ（記載がない場合は『詳細はPDF本文参照』と明記）")
    discussion_and_implications: str = Field(description="アブストラクトの結論・意義の忠実な要約")
    relevance_to_my_research: str = Field(description="脳心臓点過程モデル・非線形HRV・RC+Koopman制御への応用観点からのコメント")
    vocabulary_list: List[VocabularyItem]

# =========================================================================
# 3. PubMed 公式API 実データ取得関数
# =========================================================================
def sanitize_svg(svg_str: str, min_height: int = 220) -> str:
    if not svg_str or "<svg" not in svg_str:
        return f'<svg viewBox="0 0 600 {min_height}" xmlns="http://www.w3.org/2000/svg"><rect width="600" height="{min_height}" rx="8" fill="#f8fafc" stroke="#cbd5e1"/><text x="300" y="{min_height//2}" text-anchor="middle" fill="#475569" font-size="14">Abstract Flow Diagram</text></svg>'
    match = re.search(r"<svg[\s\S]*?</svg>", svg_str, re.IGNORECASE)
    return match.group(0) if match else svg_str


def search_pubmed_free_fulltext(query: str, max_results: int = 30) -> List[str]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
        "email": NCBI_EMAIL
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_pubmed_verified_metadata(pmid: str) -> Optional[dict]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "email": NCBI_EMAIL}
    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code != 200:
        return None

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    article = root.find(".//PubmedArticle")
    if article is None:
        return None

    title_el = article.find(".//ArticleTitle")
    title_en = " ".join("".join(title_el.itertext()).split()) if title_el is not None else ""

    journal_el = article.find(".//Journal/Title")
    journal = " ".join("".join(journal_el.itertext()).split()) if journal_el is not None else "Peer-Reviewed Journal"

    pub_year_el = article.find(".//JournalIssue/PubDate/Year")
    pub_year = pub_year_el.text.strip() if pub_year_el is not None and pub_year_el.text else ""

    abstract_texts = []
    for ab in article.findall(".//Abstract/AbstractText"):
        label = ab.attrib.get("Label", "")
        text = " ".join("".join(ab.itertext()).split())
        if text:
            abstract_texts.append(f"[{label}] {text}" if label else text)
    abstract_en = "\n\n".join(abstract_texts)

    doi = ""
    pmcid = ""
    for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = aid.attrib.get("IdType", "")
        if id_type == "doi" and aid.text:
            doi = aid.text.strip()
        elif id_type == "pmc" and aid.text:
            pmcid = aid.text.strip()

    # 実在確認のため DOI と 十分な長さの Abstract がある論文のみ採用
    if not doi or not title_en or len(abstract_en) < 200:
        return None

    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "pub_year": pub_year,
        "title_en": title_en,
        "journal": journal,
        "abstract_en": abstract_en
    }


# =========================================================================
# 4. Gemini API クライアント & アブストラクト忠実翻訳
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
            return client, candidate

    flash_models = sorted(
        [m for m in available if "flash" in m and "2.0" not in m and "2.5" not in m and "lite" not in m],
        reverse=True
    )
    if flash_models:
        return client, flash_models[0]

    return client, "gemini-3.8-flash"


def generate_abstract_digest_json(client, model_name: str, meta: dict, track: dict, today_str: str) -> dict:
    prompt = f"""
あなたは生体医工学・神経科学・非線形制御工学の専門研究者兼バイリンガル編集者です。
以下は PubMed から取得した実在の Free Full Text（Open Access）論文のメタデータおよびアブストラクト原文です。

【厳守ルール（捏造・推測の完全禁止）】
1. **アブストラクト原文に書かれている事実のみ**を忠実に日本語訳・構造化してください。アブストラクトに記載されていない実験データ、サンプル数、統計値、手法の詳細を推測・創作することは厳禁です。
2. 本文全体（Introduction〜Conclusion）はユーザーが別途PDFをダウンロードしてサイト上で解読するため、現時点ではアブストラクトの範囲を超えた本文の補完は絶対に行わないでください。
3. `diagram_svg` は背景淡色（#f8fafc）、青系（#1e3a8a, #2563eb）・ティール系（#0d9488）を基調とした概念図SVGを記述してください。
4. `vocabulary_list` には、下記の【Abstract 原文】に実際に登場する専門用語・学術動詞を8〜12個選び解説してください。

【実在論文メタデータ】
PMID: {meta['pmid']}
PMCID: {meta['pmcid'] if meta['pmcid'] else 'Publisher Open Access'}
DOI: {meta['doi']}
Year: {meta['pub_year']}
Journal: {meta['journal']}
Title: {meta['title_en']}

【Abstract 原文】
{meta['abstract_en']}
"""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AbstractDigestSchema,
            temperature=0.1
        )
    )
    parsed = json.loads(response.text)
    diagram_svg = sanitize_svg(parsed["diagram_svg"], 230)

    vocab_dict = {}
    for v in parsed.pop("vocabulary_list", []):
        vocab_dict[v["term"].lower()] = {
            "ipa": v["ipa"],
            "pos": v["pos"],
            "ja": v["ja"],
            "nuance": v["nuance"]
        }

    # 現時点では「原文アブストラクト」のみを正確に格納（PDFアップロード後に5セクション全文へ展開）
    abstract_section = [
        {
            "id": "intro",
            "title_en": "Abstract (Verified Original Text — Upload PDF for Full-Text Sections)",
            "title_ja": "アブストラクト原文（全文はリンク先からPDFを取得し画面右上の「PDF全文解読」へドロップしてください）",
            "section_svg": diagram_svg,
            "paragraphs": [
                {
                    "en": meta["abstract_en"],
                    "ja": parsed["abstract_ja"]
                }
            ]
        }
    ]

    return {
        "id": f"paper-{today_str}-{meta['pmid']}",
        "date": today_str,
        "pmid": meta["pmid"],
        "pmcid": meta["pmcid"],
        "doi_url": f"https://doi.org/{meta['doi']}",
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{meta['pmid']}/",
        "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{meta['pmcid']}/" if meta["pmcid"] else "",
        "category": track["category"],
        "category_label": track["category_label"],
        "journal": f"{meta['journal']} ({meta['pub_year']})" if meta["pub_year"] else meta["journal"],
        "impact_factor": parsed["impact_factor"],
        "doi": meta["doi"],
        "title_en": meta["title_en"],
        "title_ja": parsed["title_ja"],
        "tags": parsed["tags"],
        "summary": {
            "problem": parsed["problem"],
            "solution_and_method": parsed["solution_and_method"],
            "formula_block": parsed["formula_block"],
            "diagram_caption": parsed["diagram_caption"],
            "diagram_svg": diagram_svg,
            "key_results": parsed["key_results"],
            "discussion_and_implications": parsed["discussion_and_implications"],
            "relevance_to_my_research": parsed["relevance_to_my_research"]
        },
        "full_text": {
            "sections": abstract_section
        },
        "vocabulary": vocab_dict
    }


# =========================================================================
# 5. メイン実行フロー（5分野から1本ずつ＝計5本取得）
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
    client, model_name = get_active_client_and_model()
    print(f"Using Gemini model: {model_name}")

    generated_count = 0
    for idx, track in enumerate(DAILY_RESEARCH_TRACKS):
        print(f"\n--- [{idx+1}/{len(DAILY_RESEARCH_TRACKS)}] Track: {track['category_label']} ---")
        try:
            pmids = search_pubmed_free_fulltext(track["query"], max_results=30)
            print(f"Found {len(pmids)} candidate Free Full Text PMIDs")

            selected_meta = None
            for pmid in pmids:
                if pmid in processed_set:
                    continue
                time.sleep(0.35)
                meta = fetch_pubmed_verified_metadata(pmid)
                if meta:
                    selected_meta = meta
                    print(f"Verified paper: PMID={pmid}, PMCID={meta['pmcid']}, DOI={meta['doi']}")
                    break

            if not selected_meta:
                print(f"Warning: No new paper found for {track['category_label']}, skipping.")
                continue

            paper_data = generate_abstract_digest_json(client, model_name, selected_meta, track, today_str)
            out_path = f"data/papers/{today_str}-{selected_meta['pmid']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(paper_data, f, ensure_ascii=False, indent=2)

            manifest["papers"] = [p for p in manifest["papers"] if p["id"] != paper_data["id"]]
            manifest["papers"].insert(0, {
                "id": paper_data["id"],
                "date": today_str,
                "category": track["category"],
                "category_label": track["category_label"],
                "path": out_path,
                "title_ja": paper_data["title_ja"]
            })
            processed_set.add(selected_meta["pmid"])
            generated_count += 1

            manifest["processed_pmids"] = list(processed_set)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            if idx < len(DAILY_RESEARCH_TRACKS) - 1:
                print("Waiting 12s to respect Gemini API rate limits...")
                time.sleep(12)

        except Exception as e:
            print(f"Error processing track {track['category_label']}: {e}")
            time.sleep(12)
            continue

    if generated_count == 0:
        raise RuntimeError("Failed to generate any papers across all 5 tracks.")

    print(f"\nCompleted! Successfully generated {generated_count} verified Open Access digests today.")


if __name__ == "__main__":
    main()
