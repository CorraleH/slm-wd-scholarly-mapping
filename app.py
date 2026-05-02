# python -m pip install gradio yake requests

import gradio as gr
import yake
import requests
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_SCHOLARLY = "https://query-scholarly.wikidata.org/sparql"
OPENALEX_API = "https://api.openalex.org/works/doi:"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/DOI:"
USER_AGENT = "example (https://github.com/example; contact@example.com)"
MAILTO = "contact@example.com"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

LANGUAGE_HINTS = {
    "pt": ["que", "é", "como", "quais", "qual", "são", "pode", "sobre", "das", "dos"],
    "en": ["what", "how", "which", "does", "the", "is", "are", "about", "can"],
    "es": ["qué", "cómo", "cuáles", "los", "las", "una", "por", "son", "puede"],
    "fr": ["est", "que", "quoi", "comment", "quels", "les", "une", "des", "peut"],
    "de": ["was", "wie", "welche", "ist", "ein", "eine", "der", "die", "das"],
    "it": ["che", "cosa", "come", "quali", "una", "dei", "delle", "può"],
    "nl": ["wat", "hoe", "welke", "een", "het", "van", "zijn", "kan", "voor"],
}

history = []

# SLM loading
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto",
    )
    print(f"Model loaded on GPU ({torch.cuda.get_device_name(0)})")
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32,
    )
    print("Model loaded on CPU (slower inference)")

def generate(messages, max_tokens=1024):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

# NLP
def detect_language(text):
    text_lower = text.lower()
    scores = {}
    for lang, words in LANGUAGE_HINTS.items():
        scores[lang] = sum(1 for w in words if f" {w} " in f" {text_lower} ")
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"

def extract_keyword(question, language):
    extractor = yake.KeywordExtractor(lan=language, n=3, top=1, dedupLim=0.9, windowsSize=1)
    keywords = extractor.extract_keywords(question)
    return keywords[0][0] if keywords else question.strip()

# Wikidata
def search_entity(keyword, language):
    params = {
        "action": "wbsearchentities",
        "search": keyword,
        "language": language,
        "uselang": language,
        "type": "item",
        "limit": 5,
        "format": "json",
    }
    try:
        resp = requests.get(
            WIKIDATA_API, params=params,
            headers={"User-Agent": USER_AGENT}, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [{"error": str(e)}]
    return [
        {
            "qid": item["id"],
            "label": item.get("label", ""),
            "description": item.get("description", ""),
        }
        for item in data.get("search", [])
    ]

def search_articles(qid, limit=20):
    sparql = f"""
SELECT DISTINCT ?article ?articleLabel ?doi ?date WHERE {{
  ?article wdt:P31 wd:Q13442814 .
  ?article wdt:P921 wd:{qid} .
  OPTIONAL {{ ?article wdt:P356 ?doi . }}
  OPTIONAL {{ ?article wdt:P577 ?date . }}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en,pt,es,fr,de,it,nl" .
  }}
}}
ORDER BY DESC(?date)
LIMIT {limit}
"""
    try:
        resp = requests.get(
            SPARQL_SCHOLARLY, params={"query": sparql, "format": "json"},
            headers={"User-Agent": USER_AGENT}, timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [{"error": str(e)}]
    articles = []
    for b in data.get("results", {}).get("bindings", []):
        uri = b.get("article", {}).get("value", "")
        articles.append({
            "qid": uri.split("/")[-1] if uri else "",
            "label": b.get("articleLabel", {}).get("value", ""),
            "doi": b.get("doi", {}).get("value", ""),
            "date": b.get("date", {}).get("value", "")[:10],
        })
    return articles

# Abstracts
def abstract_openalex(doi):
    try:
        resp = requests.get(
            f"{OPENALEX_API}{doi}", params={"mailto": MAILTO},
            headers={"User-Agent": USER_AGENT}, timeout=10,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        inverted = data.get("abstract_inverted_index")
        if not inverted:
            return ""
        words = sorted(
            [(pos, word) for word, positions in inverted.items() for pos in positions],
            key=lambda x: x[0],
        )
        return " ".join(word for _, word in words)
    except Exception:
        return ""

def abstract_semanticscholar(doi):
    try:
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_API}{doi}", params={"fields": "abstract"},
            headers={"User-Agent": USER_AGENT}, timeout=10,
        )
        if resp.status_code != 200:
            return ""
        return resp.json().get("abstract") or ""
    except Exception:
        return ""

def fetch_abstract(doi):
    abstract = abstract_openalex(doi)
    if abstract:
        return abstract, "OpenAlex"
    time.sleep(0.3)
    abstract = abstract_semanticscholar(doi)
    if abstract:
        return abstract, "Semantic Scholar"
    return "", ""

# SLM synthesis
def synthesize(question, abstracts_with_doi):
    abs_text = "\n\n".join(
        f"Abstract {i+1} (DOI: {a['doi']}):\n{a['abstract']}"
        for i, a in enumerate(abstracts_with_doi)
    )
    if not abs_text.strip():
        return "No abstracts available for synthesis.", ""

    # Round 1: Synthesis
    messages_r1 = [
        {"role": "system", "content": (
            "You are a scientific summarizer. Given a question and "
            "Write in the same language as the question." 
            "Cite each claim with the DOI of the source article in parentheses."
            "abstracts from scholarly articles, write a clear and accurate "
            "answer based only on the information in the abstracts. 
        )},
        {"role": "user", "content": (
            f"Question: {question}\n\n{abs_text}\n\n"
            "Write a comprehensive answer based on these abstracts."
        )},
    ]
    response_r1 = generate(messages_r1)

    # Round 2: Verification
    messages_r2 = [
        {"role": "system", "content": (
            "Write in the same language as the question."
            "You are a scientific fact-checker. Compare the answer below "
            "with the original abstracts. Check for:\n"
            "1. Accuracy: every claim must be supported by an abstract\n"
            "2. Attribution: every claim must cite the correct DOI\n"
            "3. Completeness: key findings should not be omitted\n"
            "4. Coherence: the text should read as a unified answer\n\n"
            "If corrections are needed, output the corrected version. "
            "If the answer is accurate, output it unchanged. 
        )},
        {"role": "user", "content": (
            f"Question: {question}\n\n"
            f"Generated answer:\n{response_r1}\n\n"
            f"Original abstracts:\n{abs_text}\n\n"
            "Verify and correct if needed."
        )},
    ]
    response_r2 = generate(messages_r2)

    return response_r1, response_r2

# Main pipeline
def slmwd(question):
    if not question or not question.strip():
        return "", "", "", gr.Tabs(selected="papers"), format_history()

    language = detect_language(question)
    keyword = extract_keyword(question, language)
    history.append(f"{question}  [{language}, {keyword}]")

    lines = []
    lines.append(f"Language: {language}")
    lines.append(f"Keyword: {keyword}")
    lines.append("")

    entities = search_entity(keyword, language)
    if not entities:
        lines.append("No entities found on Wikidata.")
        return "\n".join(lines), "", "", gr.Tabs(selected="papers"), format_history()
    if "error" in entities[0]:
        lines.append(f"Entity search error: {entities[0]['error']}")
        return "\n".join(lines), "", "", gr.Tabs(selected="papers"), format_history()

    lines.append("Entities:")
    for e in entities:
        lines.append(f"  {e['qid']}  {e['label']}  ({e['description']})")
    lines.append("")

    top_qid = entities[0]["qid"]
    top_label = entities[0]["label"]
    lines.append(f"Searching scholarly articles for: {top_label} ({top_qid})")
    lines.append(f"Endpoint: query-scholarly.wikidata.org")
    lines.append("")

    articles = search_articles(top_qid)
    if not articles:
        lines.append("No scholarly articles found.")
        return "\n".join(lines), "", "", gr.Tabs(selected="papers"), format_history()
    if "error" in articles[0]:
        lines.append(f"SPARQL error: {articles[0]['error']}")
        return "\n".join(lines), "", "", gr.Tabs(selected="papers"), format_history()

    lines.append(f"Articles: {len(articles)}")
    lines.append("")

    for i, art in enumerate(articles, 1):
        lines.append(f"{i}. [{art['qid']}] {art['label']}")
        if art["doi"]:
            lines.append(f"   DOI: {art['doi']}")
        if art["date"]:
            lines.append(f"   Date: {art['date']}")
        lines.append("")

    # Fetch abstracts
    abstract_lines = []
    abstract_lines.append(f"Abstracts for: {keyword}")
    abstract_lines.append("")

    abstracts_for_slm = []
    count = 0
    for art in articles:
        if not art["doi"]:
            continue
        abstract, source = fetch_abstract(art["doi"])
        count += 1
        abstract_lines.append(f"{count}. {art['label']}")
        abstract_lines.append(f"   DOI: {art['doi']}")
        if abstract:
            abstract_lines.append(f"   Source: {source}")
            abstract_lines.append(f"   Abstract: {abstract}")
            abstracts_for_slm.append({"doi": art["doi"], "abstract": abstract})
        else:
            abstract_lines.append(f"   Abstract: not available")
        abstract_lines.append("")
        time.sleep(0.2)

    abstract_lines.append(f"Total: {count} DOIs queried")
    abstract_lines.append(f"Abstracts retrieved: {len(abstracts_for_slm)}")

    # SLM synthesis + verification
    answer_lines = []
    if abstracts_for_slm:
        answer_lines.append(f"Model: {MODEL_ID}")
        answer_lines.append(f"Abstracts used: {len(abstracts_for_slm)}")
        answer_lines.append("")

        response_r1, response_r2 = synthesize(question, abstracts_for_slm)

        answer_lines.append("--- Round 1: Synthesis ---")
        answer_lines.append("")
        answer_lines.append(response_r1)
        answer_lines.append("")
        answer_lines.append("--- Round 2: Verification ---")
        answer_lines.append("")
        answer_lines.append(response_r2)
    else:
        answer_lines.append("No abstracts available for SLM synthesis.")

    return (
        "\n".join(lines),
        "\n".join(abstract_lines),
        "\n".join(answer_lines),
        gr.Tabs(selected="papers"),
        format_history(),
    )

def format_history():
    if not history:
        return ""
    lines = []
    for i, q in enumerate(history, 1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)

# UI 
with gr.Blocks(
    theme=gr.themes.Default(primary_hue=gr.themes.colors.blue),
    css="""
        .gradio-container {
            font-size: 14pt; color: #000; background-color: #fff;
            width: 100%; margin: auto; padding-top: 20vh; border: none;
        }
        * {font-size: 14pt !important; color: #000 !important;}
        label {background-color: #fff; border: none;}
        textarea {background-color: #f8f8ff !important; border: none; color: #000 !important;}
        textarea:focus {background-color: #f8f8ff !important; color: #000 !important; outline: none;}
        div {border: none; background-color: #fff; padding-left: 0 !important;}
        footer {display: none !important;}
        button {
            background-color: #1976d2 !important; border: none;
            height: 34px; padding: 0 14px; color: #fff !important;
            display: flex; align-items: center; justify-content: center;
        }
        button:hover {
            background-color: #1565c0 !important; color: #fff !important;
        }
        #ask-btn {width: 100px !important; display: flex; margin-left: 0 !important; min-width: unset !important;}
        #history textarea {font-size: 11pt !important; color: #666 !important; background-color: #fff !important;}
        [role="tablist"] {
            border: none !important; box-shadow: none !important; border-bottom: none !important;
        }
        [role="tab"] {
            background-color: #fff !important; color: #999 !important;
            border: none !important; box-shadow: none !important;
            border-bottom: 2px solid transparent !important;
            margin-right: 24px !important; padding: 8px 4px !important;
        }
        [role="tab"]:hover {
            color: #000 !important; border-bottom: 2px solid transparent !important;
            background-color: #fff !important;
        }
        [role="tab"][aria-selected="true"] {
            color: #000 !important; border-bottom: 2px solid #1976d2 !important;
        }
        [role="tabpanel"] {
            border: none !important; box-shadow: none !important; padding-top: 20px !important;
        }
    """
) as demo:
    with gr.Tabs(elem_id="tabs") as tabs:
        with gr.Tab("Question", id="question"):
            question_box = gr.Textbox(label="Ask to Wikidata", lines=3)
            ask_btn = gr.Button("Ask", size="sm", elem_id="ask-btn")
            history_box = gr.Textbox(
                label="History", lines=4, interactive=False, elem_id="history",
            )
        with gr.Tab("Papers", id="papers"):
            result_box = gr.Textbox(lines=14, interactive=False, show_label=False)
        with gr.Tab("Abstracts", id="abstracts"):
            abstract_box = gr.Textbox(lines=20, interactive=False, show_label=False)
        with gr.Tab("Answer", id="answer"):
            answer_box = gr.Textbox(lines=20, interactive=False, show_label=False)
    ask_btn.click(
        fn=slmwd,
        inputs=question_box,
        outputs=[result_box, abstract_box, answer_box, tabs, history_box],
    )
demo.launch(share=True)
