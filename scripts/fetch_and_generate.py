import os
import json
import time
import datetime
import html
import requests
import xml.etree.ElementTree as ET
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NCBI_EMAIL = "research-digest@example.com"

# =========================================================================
# 1. 毎日取得する5分野の研究トラック＆臨床横断レビューテーマ（OA限定）
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

CLINICAL_REVIEW_TOPICS = [
    {
        "domain_label": "認知症・神経変性疾患の早期バイオマーカーと自律神経・青斑核連関",
        "query": '("locus coeruleus"[Title/Abstract] OR "prodromal Alzheimer"[Title/Abstract] OR "mild cognitive impairment"[Title/Abstract]) AND ("autonomic"[Title/Abstract] OR "heart rate variability"[Title/Abstract]) AND "free full text"[filter]'
    },
    {
        "domain_label": "てんかん・難治性神経疾患における脳心臓連関と迷走神経刺激制御",
        "query": '("epilepsy"[Title/Abstract] OR "vagus nerve stimulation"[Title/Abstract]) AND ("heart rate variability"[Title/Abstract] OR "brain-heart"[Title/Abstract]) AND "free full text"[filter]'
    }
]

# =========================================================================
# 2. Pydantic スキーマ（デイリー論文5本 ＋ 臨床統合レビュー1本）
# =========================================================================
class VocabularyItem(BaseModel):
    term: str = Field(description="英語アブストラクト内に実際に登場する学術用語")
    ipa: str = Field(description="国際音声記号")
    pos: str = Field(description="品詞（noun, verb, adj 等）")
    ja: str = Field(description="学術的な日本語訳")
    nuance: str = Field(description="論文読解・執筆における用法解説")

class SinglePaperDigest(BaseModel):
    pmid: str = Field(description="入力されたPMIDをそのまま返す")
    title_ja: str = Field(description="英語論文タイトルの正確な日本語訳")
    abstract_ja: str = Field(description="英語アブストラクト原文の忠実な日本語訳（推測・補完禁止）")
    impact_factor: float = Field(description="掲載誌の直近インパクトファクター概算値（不明時は3.5）")
    tags: List[str] = Field(description="専門タグ4個")
    problem: str = Field(description="アブストラクトに記載された研究背景・課題")
    solution_and_method: str = Field(description="アブストラクトに記載された手法・解析方法")
    formula_block: str = Field(description="関連する基本数式または指標のLaTeX表記（例: $$...$$）")
    key_results: str = Field(description="アブストラクトに明記されている結果・数値のみ")
    discussion_and_implications: str = Field(description="アブストラクトの結論・意義")
    relevance_to_my_research: str = Field(description="脳心臓点過程・非線形HRV・RC+Koopman制御への応用ヒント")
    vocabulary_list: List[VocabularyItem] = Field(description="アブストラクト内の重要用語5〜8個")

class BatchDigestResponse(BaseModel):
    digests: List[SinglePaperDigest]

class ClinicalReviewSynthesis(BaseModel):
    title_ja: str = Field(description="臨床横断レビューの日本語タイトル")
    title_en: str = Field(description="臨床横断レビューの英語タイトル")
    tags: List[str] = Field(description="臨床・バイオマーカー関連タグ4〜5個")
    problem: str = Field(description="現在の臨床現場における未解決課題・早期診断の壁（日本語）")
    solution_and_method: str = Field(description="対象論文群で用いられている計測・解析アプローチの統合比較（日本語）")
    formula_block: str = Field(description="関連する自律神経・点過程・バイオマーカー評価指標のLaTeX数式")
    key_results: str = Field(description="対象論文群から得られた主要な臨床エビデンス・統計的知見（日本語）")
    discussion_and_implications: str = Field(description="複数論文を横断した臨床的意義と今後の展望（日本語）")
    relevance_to_my_research: str = Field(description="数理モデリング・非線形HRV・閉ループ神経制御への応用指針（日本語）")

# =========================================================================
# 3. PubMed 公式API 実データ取得＆SVG生成
# =========================================================================
def build_clean_svg(category_label: str, pmid: str, doi: str) -> str:
    safe_cat = html.escape(category_label)
    safe_doi = html.escape(doi[:42])
    return f'''<svg viewBox="0 0 600 180" xmlns="http://www.w3.org/2000/svg">
  <rect width="600" height="180" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <rect x="24" y="24" width="165" height="56" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="106" y="48" text-anchor="middle" fill="#1e3a8a" font-size="12" font-weight="bold">PubMed / PMC OA</text>
  <text x="106" y="66" text-anchor="middle" fill="#334155" font-size="11">PMID: {pmid}</text>
  <path d="M 195 52 L 225 52" stroke="#0d9488" stroke-width="2"/>
  <rect x="232" y="24" width="175" height="56" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1.5"/>
  <text x="319" y="48" text-anchor="middle" fill="#115e59" font-size="12" font-weight="bold">{safe_cat}</text>
  <text x="319" y="66" text-anchor="middle" fill="#334155" font-size="10">Verified Abstract Digest</text>
  <path d="M 413 52 L 443 52" stroke="#0d9488" stroke-width="2"/>
  <rect x="450" y="24" width="126" height="56" rx="8" fill="#1e3a8a"/>
  <text x="513" y="48" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="bold">PDF Full-Text</text>
  <text x="513" y="65" text-anchor="middle" fill="#bfdbfe" font-size="10">Drop PDF to Decode</text>
  <rect x="24" y="102" width="552" height="54" rx="6" fill="#ffffff" stroke="#e2e8f0"/>
  <text x="38" y="124" fill="#475569" font-size="11" font-weight="bold">Verified DOI Source:</text>
  <text x="38" y="143" fill="#2563eb" font-size="11">https://doi.org/{safe_doi}</text>
</svg>'''


def search_pubmed_free_fulltext(query: str, max_results: int = 25) -> List[str]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
        "email": NCBI_EMAIL
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_pubmed_verified_metadata(pmid: str) -> Optional[dict]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "email": NCBI_EMAIL}
    resp = requests.get(url, params=params, timeout=15)
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
# 4. Gemini API クライアント（503混雑時の自動モデル切り替え対応）
# =========================================================================
def get_candidate_models(client) -> List[str]:
    preferred = [
        "gemini-3.8-flash-lite",
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
                if "gemini" in name and "embedding" not in name and "imagen" not in name and "2.0" not in name and "2.5" not in name:
                    available.append(name)
    except Exception as e:
        print(f"Model list check skipped: {e}")

    ordered = [m for m in preferred if m in available]
    for m in sorted(available, reverse=True):
        if m not in ordered and "pro" not in m:
            ordered.append(m)

    return ordered if ordered else ["gemini-3.8-flash-lite", "gemini-3.8-flash"]


def call_gemini_with_fallback(client, models_to_try: List[str], prompt: str, schema_cls):
    last_error = None
    for model_name in models_to_try:
        try:
            print(f"Calling Gemini model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_cls,
                    temperature=0.1
                )
            )
            return schema_cls.model_validate_json(response.text)
        except Exception as e:
            print(f"Model {model_name} unavailable or failed ({e}). Trying next...")
            last_error = e
            time.sleep(2)
    raise RuntimeError(f"All candidate Gemini models failed. Last error: {last_error}")


# =========================================================================
# 5. メイン実行フロー（デイリー5本 ＋ 臨床統合レビュー1本）
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
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            api_version="v1beta",
            timeout=60000,
            retry_options=types.HttpRetryOptions(attempts=2, initial_delay=2.0, max_delay=8.0)
        )
    )
    models_to_try = get_candidate_models(client)

    # 1. 5分野のデイリー論文収集
    candidates = []
    for idx, track in enumerate(DAILY_RESEARCH_TRACKS):
        print(f"[{idx+1}/5] Searching PubMed OA for: {track['category_label']}")
        try:
            pmids = search_pubmed_free_fulltext(track["query"], max_results=25)
            for pmid in pmids:
                if pmid in processed_set or any(c["meta"]["pmid"] == pmid for c in candidates):
                    continue
                time.sleep(0.35)
                meta = fetch_pubmed_verified_metadata(pmid)
                if meta:
                    candidates.append({"track": track, "meta": meta})
                    print(f"  -> Found PMID={pmid} | DOI={meta['doi']}")
                    break
        except Exception as e:
            print(f"  -> PubMed search error for {track['category_label']}: {e}")

    if candidates:
        papers_block = [
            f"=== PAPER PMID: {c['meta']['pmid']} | TRACK: {c['track']['category_label']} ===\n"
            f"Journal: {c['meta']['journal']} ({c['meta']['pub_year']})\n"
            f"DOI: {c['meta']['doi']}\n"
            f"Title: {c['meta']['title_en']}\n"
            f"Abstract:\n{c['meta']['abstract_en']}\n"
            for c in candidates
        ]
        prompt = f"""
あなたは生体医工学・神経科学・非線形制御工学の専門研究者兼バイリンガル編集者です。
以下は PubMed から取得した実在する {len(candidates)} 本の Free Full Text 論文です。
アブストラクトに記載されている事実のみに基づき、全 {len(candidates)} 本の忠実な日本語訳と構造化サマリーを作成してください（推測・創作厳禁）。

{chr(10).join(papers_block)}
"""
        batch_res = call_gemini_with_fallback(client, models_to_try, prompt, BatchDigestResponse)
        digest_by_pmid = {d.pmid: d for d in batch_res.digests}

        for c in candidates:
            meta = c["meta"]
            track = c["track"]
            d = digest_by_pmid.get(meta["pmid"])
            if not d:
                continue

            diagram_svg = build_clean_svg(track["category_label"], meta["pmid"], meta["doi"])
            vocab_dict = {
                v.term.lower(): {"ipa": v.ipa, "pos": v.pos, "ja": v.ja, "nuance": v.nuance}
                for v in d.vocabulary_list
            }
            paper_data = {
                "id": f"paper-{today_str}-{meta['pmid']}",
                "entry_type": "daily_paper",
                "date": today_str,
                "pmid": meta["pmid"],
                "pmcid": meta["pmcid"],
                "doi_url": f"https://doi.org/{meta['doi']}",
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{meta['pmid']}/",
                "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{meta['pmcid']}/" if meta["pmcid"] else "",
                "category": track["category"],
                "category_label": track["category_label"],
                "journal": f"{meta['journal']} ({meta['pub_year']})" if meta["pub_year"] else meta["journal"],
                "impact_factor": d.impact_factor,
                "doi": meta["doi"],
                "title_en": meta["title_en"],
                "title_ja": d.title_ja,
                "tags": d.tags,
                "summary": {
                    "problem": d.problem,
                    "solution_and_method": d.solution_and_method,
                    "formula_block": d.formula_block,
                    "diagram_caption": f"{track['category_label']} — Verified Open Access Source",
                    "diagram_svg": diagram_svg,
                    "key_results": d.key_results,
                    "discussion_and_implications": d.discussion_and_implications,
                    "relevance_to_my_research": d.relevance_to_my_research
                },
                "full_text": {
                    "sections": [
                        {
                            "id": "intro",
                            "title_en": "Abstract (Verified Original Text — Drop PDF to Decode Full Text)",
                            "title_ja": "アブストラクト原文（全文はリンク先からPDFを取得し画面右下の「PDF全文解読」へドロップしてください）",
                            "section_svg": diagram_svg,
                            "paragraphs": [{"en": meta["abstract_en"], "ja": d.abstract_ja}]
                        }
                    ]
                },
                "vocabulary": vocab_dict
            }

            out_path = f"data/papers/{today_str}-{meta['pmid']}.json"
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
            processed_set.add(meta["pmid"])

    # 2. 臨床最前線統合レビュー（複数論文を統合した臨床レビューを1本生成）
    topic = CLINICAL_REVIEW_TOPICS[now.timetuple().tm_yday % len(CLINICAL_REVIEW_TOPICS)]
    print(f"\n[Clinical Review] Searching OA papers for: {topic['domain_label']}")
    rev_pmids = search_pubmed_free_fulltext(topic["query"], max_results=20)
    rev_metas = []
    for pmid in rev_pmids:
        time.sleep(0.35)
        meta = fetch_pubmed_verified_metadata(pmid)
        if meta:
            rev_metas.append(meta)
        if len(rev_metas) >= 3:
            break

    if rev_metas:
        rev_papers_text = "\n\n".join([
            f"[{i+1}] PMID: {m['pmid']} | DOI: {m['doi']} | Journal: {m['journal']} ({m['pub_year']})\nTitle: {m['title_en']}\nAbstract: {m['abstract_en']}"
            for i, m in enumerate(rev_metas)
        ])
        rev_prompt = f"""
あなたは臨床神経科学および生体医工学のシニアリサーチャーです。
以下の実在する {len(rev_metas)} 本の Open Access 査読済み論文のみに基づき、「{topic['domain_label']}」に関する臨床最前線統合レビューを作成してください（記載のない論文や数値の創作厳禁）。

{rev_papers_text}
"""
        rev_res = call_gemini_with_fallback(client, models_to_try, rev_prompt, ClinicalReviewSynthesis)
        sources_list_text = "\n".join([f"- {m['title_en']} (PMID: {m['pmid']}, DOI: https://doi.org/{m['doi']})" for m in rev_metas])
        rev_svg = build_clean_svg("臨床最前線統合レビュー", rev_metas[0]["pmid"], rev_metas[0]["doi"])

        review_data = {
            "id": f"rev-{today_str}",
            "entry_type": "clinical_review",
            "date": today_str,
            "category": "clinical",
            "category_label": "臨床最前線統合レビュー",
            "journal": f"Integrated Clinical Review ({len(rev_metas)} OA Studies)",
            "impact_factor": 6.5,
            "doi": rev_metas[0]["doi"],
            "title_en": rev_res.title_en,
            "title_ja": rev_res.title_ja,
            "tags": rev_res.tags,
            "summary": {
                "problem": rev_res.problem,
                "solution_and_method": rev_res.solution_and_method,
                "formula_block": rev_res.formula_block,
                "diagram_caption": topic["domain_label"],
                "diagram_svg": rev_svg,
                "key_results": rev_res.key_results,
                "discussion_and_implications": rev_res.discussion_and_implications,
                "relevance_to_my_research": f"【統合対象の実在Open Access論文】\n{sources_list_text}\n\n{rev_res.relevance_to_my_research}"
            },
            "full_text": {
                "sections": [
                    {
                        "id": "intro",
                        "title_en": f"Integrated Evidence ({len(rev_metas)} Verified Open Access Papers)",
                        "title_ja": "統合エビデンス（引用元アブストラクト原文）",
                        "section_svg": rev_svg,
                        "paragraphs": [
                            {"en": f"[{m['title_en']}]\n{m['abstract_en']}", "ja": rev_res.key_results}
                            for m in rev_metas
                        ]
                    }
                ]
            },
            "vocabulary": {}
        }

        rev_path = f"data/reviews/{today_str}.json"
        with open(rev_path, "w", encoding="utf-8") as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)

        manifest["reviews"] = [r for r in manifest["reviews"] if r["id"] != review_data["id"]]
        manifest["reviews"].insert(0, {
            "id": review_data["id"],
            "date": today_str,
            "path": rev_path,
            "title_ja": review_data["title_ja"]
        })

    manifest["processed_pmids"] = list(processed_set)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nCompleted! Updated daily papers and clinical frontline review.")


if __name__ == "__main__":
    main()
