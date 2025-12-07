AI Legal Toolkit (OpenAI + Streamlit)
An AI-powered legal research and courtroom simulation tool built with Python, Streamlit, LangChain, and OpenAI models.
 Supports case-based argument generation, retrieval-augmented legal Q&A, and courtroom dialogue simulation using FIR documents and Indian law (BNS + Constitution).

Features
| Module                   | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| Courtroom Simulation     | Upload an FIR and get a full simulated trial with Judge and Lawyers          |
| Argument Generator (RAG) | Upload a PDF and generate client arguments, counterarguments, legal strategy |
| General Legal QA         | Ask legal questions grounded only in Constitution and BNS                    |
| Secure Login System      | Users sign in or register (stored in YAML)                                   |



Tech Stack
Python 3.10+


Streamlit


LangChain


OpenAI (ChatOpenAI, OpenAIEmbeddings)


ChromaDB


YAML configuration


Local CSV-based law documents



Quick Start
1. Clone
git clone https://github.com/<your-username>/<repo>.git
cd ai-legal-toolkit

2. Create virtual environment
python -m venv venv

Activate:
Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

3. Install requirements
pip install -r requirements.txt


Secrets (important)
Create:
.streamlit/secrets.toml

Inside it:
OPENAI_API_KEY="sk-your-key"

Do not commit this file.

Authentication Config
Create:
config.yaml

Example template:
credentials:
  usernames: {}
cookie:
  expiry_days: 30
  key: 'your_cookie_key'
  name: 'legal_cookie'
preauthorized:
  emails: []

Do not commit this file.

Project Structure
ai-legal-toolkit/
    combined.py
    requirements.txt
    data/
        constitution.csv
        bns.csv
    .streamlit/
        secrets.toml
    config.yaml


Data Files
Place these files inside the data directory:
constitution.csv


bns.csv


Example format:
Article,Description
Article 21,Protection of Life and Personal Liberty

Section,Description
420,Cheating and dishonestly inducing delivery of property


Run the Application
streamlit run combined.py

Open in browser:
http://localhost:8501


Architecture
The system performs:
PDF/FIR text extraction


Chunking (RecursiveCharacterTextSplitter)


Embeddings (OpenAIEmbeddings)


Vector indexing with Chroma


Retrieval Augmented Generation using ChatOpenAI



Security Notes
No API keys should appear in code


secrets.toml is ignored through .gitignore


YAML config is ignored


Rotate OpenAI keys for production use



Deployment (Streamlit Cloud)
Push repository (without secrets)


Open Streamlit Cloud settings


Add secrets:


OPENAI_API_KEY="sk-key"

Ensure data folder contains CSVs


Deploy



Troubleshooting
OpenAIAuthenticationError
Check secrets.toml
File not found: constitution.csv
Place files under /data
Model not available
Change default model:
get_llm(model_name="gpt-4o-mini")


Future Improvements
Indian case law database integration


Automated citations


IPC support


Evidence-based reasoning

