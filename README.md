# 🤖 Hirebot AI Agent

**Hirebot** is an intelligent AI agent designed to revolutionize the job search process. By leveraging advanced language models and real-time job data, Hirebot seamlessly matches user resumes with relevant job postings, streamlining the path to employment.

---

## 🚀 Features

- 🔍 **AI-Powered Resume Parsing**  
  Automatically extracts skills and experience from uploaded resumes using advanced language models.

- 🌐 **Real-Time Job Scraping**  
  Retrieves the latest job listings from LinkedIn using Apify's scraping capabilities.

- 🧠 **Intelligent Matching Algorithm**  
  Compares user profiles with job descriptions to compute relevance scores and provide the best matches.

- 🖥️ **User-Friendly Web Interface**  
  Built with Flask, enabling easy interaction through a clean and simple UI.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask  
- **AI Integration**: Together AI's Meta LLaMA-3.3 Turbo  
- **Job Scraping**: Apify LinkedIn Jobs Scraper  
- **Resume Parsing**: pdfplumber  
- **Caching**: Python `functools.lru_cache`

---

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PrabhaSuresh/Hirebot-AI.git
cd Hirebot-AI/agent_1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
TOGETHER_API_KEY=your_together_ai_api_key
APIFY_API_TOKEN=your_apify_api_token
```

---

## 🧪 Usage

### 1. Run the Application

```bash
python hirebot.py
```

### 2. Access the Interface

Open your browser and go to:

```
http://localhost:5000
```

### 3. Use the Agent

- Upload your resume (PDF format)
- Enter your job preferences:
  - Job Role
  - Location
  - Type: Full-time / Internship
  - Work Mode: Remote / Onsite / Hybrid
- Submit to receive tailored job recommendations

---

## 📊 Example Output

```
✔ Extracted Skills: Python, Machine Learning, Data Analysis
✔ Extracted Experience: 3 years in Data Science roles

🔍 Found 10 job postings for: "Data Scientist"
📊 Top Match: XYZ Corp | Score: 95%
📊 Second Match: ABC Inc | Score: 89%
...
```

---

## 📁 Project Structure

```
agent_1/
├── hirebot.py           # Main Flask application
├── templates/
│   └── index.html       # Web UI template
├── requirements.txt     # Python dependencies
```

---

## 🤝 Contributing

Contributions are welcome!  
If you'd like to add features, fix bugs, or improve performance:

1. Fork the repo
2. Create a new branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute it.

---

## 🧠 About

Made with 💡 by [Prabha Suresh](https://github.com/PrabhaSuresh)  
🔗 [Explore the full project](https://github.com/PrabhaSuresh/Hirebot-AI)

