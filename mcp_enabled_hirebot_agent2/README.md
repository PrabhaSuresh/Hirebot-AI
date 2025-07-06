

# 📁 Hirebot AI – MCP Enabled (`mcp_enabled_hirebot_agent2`)

🚀 An advanced, modular AI job matching system powered by **Model Context Protocol (MCP)**, leveraging **LLMs** and real-time scraping to deliver personalized LinkedIn job recommendations.

---

## 📌 Project Highlights

- 🧠 MCP-based agent coordination
- 📄 Resume parsing from PDFs using SpaCy + PyMuPDF
- 🌐 Real-time job listing fetch via Apify LinkedIn Scraper
- 🤖 Job scoring using Meta LLaMA 3 via Together AI
- 🧩 Modular design with reusable shared models
- 🖥️ Web UI via `templates/index.html` (Flask backend)

---

## 📂 Folder Structure

```
mcp_enabled_hirebot_agent2/
│
├── client_server.py           # Flask backend to serve the UI and handle user input
├── linkedin-scraper.py        # Scraper agent that pulls LinkedIn jobs using Apify
├── scoring-calculator.py      # Scoring agent that uses LLM to rank job matches
├── shared_models.py           # Common utilities or schema across agents
├── templates/
│   └── index.html             # Simple web frontend for resume upload
├── .env                       # Environment variables (API keys, config)
├── .gitignore
├── pyproject.toml             # Project dependencies/config (used by `uv` or `hatch`)
├── uv.lock                    # Lock file for reproducible environments
└── README.md                  # You're here!
```

---

## 🛠️ Tech Stack

| Component              | Technology                              |
|------------------------|------------------------------------------|
| Backend/API            | Flask + Python                          |
| Parsing                | SpaCy, PyMuPDF                          |
| Scraping               | Apify (LinkedIn data)                   |
| AI/LLM Scoring         | Meta LLaMA 3 via Together.ai            |
| Coordination Protocol  | Model Context Protocol (MCP)           |
| Frontend               | HTML (Jinja via Flask)                  |

---

## 🔁 Workflow

1. **User uploads resume** via `index.html` → `client_server.py`.
2. **Resume Parsing** extracts skills & experience.
3. **LinkedIn Scraper Agent** fetches relevant job listings.
4. **Scoring Agent** sends data to Together.ai LLM → returns ranked jobs.
5. **Client displays** top results to the user.

Agents share state through a **shared context** using MCP.

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/PrabhaSuresh/Hirebot-AI.git
cd Hirebot-AI/mcp_enabled_hirebot_agent2

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> Make sure to configure your `.env` file with:
```
TOGETHER_API_KEY=your_key
APIFY_API_TOKEN=your_token
```

---

## ✅ Run the App

```bash
python client_server.py
```

Then visit: [http://localhost:5000](http://localhost:5000)

---

## 🌱 Future Enhancements

- 📄 Cover Letter Generator Agent  
- 🧠 AI Feedback Loop for Job Preferences  
- 🗂️ Skill Matching & Gap Detection

---

## 📃 License

MIT License

---

## 🙌 Acknowledgements

- [Together.ai](https://www.together.ai/)
- [Apify](https://apify.com/)
- [LangChain](https://www.langchain.com/)
- [Model Context Protocol (MCP)](https://github.com/mcprotocol/mcp)
