# Legal Insight Assistant
**An Offline RAG-based Legal Assistant powered by Llama-3 & LangChain**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-black?style=for-the-badge&logo=ollama&logoColor=white)
![Llama 3](https://img.shields.io/badge/Model-Llama--3-blueviolet?style=for-the-badge)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)

## Overview
Legal Insight AI is a Retrieval-Augmented Generation (RAG) application designed to analyze Indian Legal Documents (IPC/BNS) and provide accurate, citation-backed answers.

This system leverages **Meta's Llama-3 (8B)** locally to interpret complex legal statutes without sending sensitive data to the cloud.

## Architecture
* **Orchestration:** LangChain
* **LLM (Inference):** Meta Llama-3 (via Ollama)
* **Vector Database:** FAISS (CPU Optimized)
* **Ingestion Strategy:** Recursive Character Split (Chunk Size: 1000, Overlap: 200)
* **Frontend:** Streamlit

## Key Features
* **100% Offline Privacy:** Uses a local LLM runner (Ollama), ensuring no client data or legal queries leave the machine.
* **High-Fidelity Retrieval:** Tuned chunking strategies (1000 tokens) allow the model to capture full context of "Crime" and "Punishment" sections simultaneously.
* **Citation Enforcement:** Custom prompt engineering restricts the model from hallucinating, forcing it to cite specific sections (e.g., "Section 302 IPC") from the source text.

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/dysrea/legal-insight-ai
cd legal-insight-ai
```
**2. Install dependencies**
```bash
pip install -r requirements.txt
```
**3. Setup Ollama**
* Download and install [Ollama](ollama.com)
* Pull the Llama-3 model:
```bash
ollama pull llama3
```

## Usage

**1. Ingest Data**
Place your legal PDF (e.g., `ipc.pdf`) in the `data/` folder and run:
```bash
python src/ingest.py
```
**2. Run the app**
```bash
streamlit run src/app.py
```

## Engineering Decisions
* **Why Offline RAG?** To comply with data privacy standards in legal tech, an "air-gapped" architecture was chosen. All inference happens on-device using quantized models.
* **Model Selection:** Selected **Llama-3** for its superior reasoning capabilities in complex textual analysis compared to smaller models.
* **Optimization:** Utilized **FAISS (CPU)** for vector search to reserve maximum VRAM for the Llama-3 inference engine.