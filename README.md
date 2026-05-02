# SLM-Wikidata-Scholarly-Mapping

A prototype to introduce students and academics to Wikidata. It offers an end-to-end open solution for answering questions based on scientific articles.

## What It Does

- Takes a question from the user
- Finds related entities on Wikidata
- Searches for scholarly articles
- Fetches abstracts from OpenAlex or Semantic Scholar
- Uses a Small Language Model (SLM) to synthesize answers
- Verifies the answer for accuracy and citations
- Multilingual (en, pt, es, fr, de, it, nl)

## Tech Stack

- Python
- Gradio (UI)
- Transformers (Qwen2.5-3B-Instruct)
- YAKE (keyword extraction)
- Wikidata API
- OpenAlex API
- Semantic Scholar API

## Installation

1. Clone the repository
2. Install dependencies
3. Run the app

## Usage

```bash
python app.py
