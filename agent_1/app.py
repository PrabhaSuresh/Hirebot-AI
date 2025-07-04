from dotenv import load_dotenv
from together import Together
from apify_client import ApifyClient
from flask import Flask, request, render_template
import pdfplumber
from io import BytesIO
import json
import re
import os
import uuid
from functools import lru_cache
from typing import List, Dict, Any

load_dotenv()

COUNTRY_MAPPING = {
    "united states": "United States",
    "new york": "United States",
    "california": "United States",
    "san francisco": "United States",
    "united kingdom": "United Kingdom",
    "london": "United Kingdom",
    "canada": "Canada",
    "toronto": "Canada",
    "india": "India",
    "bangalore": "India",
    "mumbai": "India",
    "delhi": "India",
}

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip()) if text else ''

def init_together_client():
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set. Please set it in .env or environment.")
    return Together(api_key=api_key)

def extract_skills(cv_text: str, client: Together) -> List[str]:
    prompt = f"""
Extract a list of professional skills from the following CV text. Return only the skills as a comma-separated list (e.g., Python, SQL, communication).

CV Text:
{cv_text}
"""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )
    skills_text = response.choices[0].message.content.strip()
    return [skill.strip() for skill in skills_text.split(',') if skill.strip()]

def extract_experience(cv_text: str, client: Together) -> Dict[str, Any]:
    prompt = f"""
Analyze the following CV text and extract the total years of professional experience and a list of job roles (e.g., Software Engineer, Data Analyst). Return the result as a JSON object with:
- 'years_experience': Total years of professional experience (integer, estimate if not explicit).
- 'job_roles': List of job titles/roles.

CV Text:
{cv_text}
"""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )
    try:
        result = json.loads(response.choices[0].message.content.strip())
        return result
    except:
        return {"years_experience": 0, "job_roles": []}

@lru_cache(maxsize=100)
def extract_job_experience(job_description: str, client: Together) -> Dict[str, Any]:
    prompt = f"""
Analyze the following job description and extract the required years of experience (integer, estimate if not explicit) and preferred job roles (e.g., Software Engineer, Data Analyst). Return the result as a JSON object with:
- 'required_years': Required years of experience.
- 'preferred_roles': List of preferred job titles/roles.

Job Description:
{job_description}
"""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )
    try:
        result = json.loads(response.choices[0].message.content.strip())
        return result
    except:
        return {"required_years": 0, "preferred_roles": []}

def calculate_experience_score(user_experience: Dict[str, Any], job_experience: Dict[str, Any], client: Together) -> int:
    prompt = f"""
Compare the user's experience with the job's experience requirements and assign an experience score from 0 to 100. Consider:
- User's years of experience: {user_experience['years_experience']}
- User's job roles: {', '.join(user_experience['job_roles'])}
- Job's required years: {job_experience['required_years']}
- Job's preferred roles: {', '.join(job_experience['preferred_roles'])}

Return only the score as a number.
"""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.7
    )
    score_text = response.choices[0].message.content.strip()
    try:
        return int(score_text)
    except:
        return 0

@lru_cache(maxsize=100)
def extract_job_skills(job_description: str, client: Together) -> List[str]:
    prompt = f"""
Extract a list of required skills from the following job description. Return only the skills as a comma-separated list (e.g., Python, SQL, machine learning).

Job Description:
{job_description}
"""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )
    skills_text = response.choices[0].message.content.strip()
    return [skill.strip() for skill in skills_text.split(',') if skill.strip()]

def match_skills(user_skills: List[str], job_description: str, client: Together) -> int:
    prompt = f"""
Compare the user's skills with the job description and assign a relevance score from 0 to 100. Return only the score as a number.

User Skills: {', '.join(user_skills)}
Job Description: {job_description}
"""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.7
    )
    score_text = response.choices[0].message.content.strip()
    try:
        return int(score_text)
    except:
        return 0

def parse_cv(file):
    text = ''
    if file.filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''
    else:  # Assume text file
        text = file.read().decode('utf-8')
    return text

def scrape_linkedin_jobs(search_term, location, job_type, selection_type='Onsite', max_jobs=10):
    api_token = os.getenv("APIFY_API_TOKEN")
    if not api_token:
        raise ValueError("APIFY_API_TOKEN not set. Please set it in .env or environment.")
    client = ApifyClient(api_token)

    search_query = search_term
    if job_type == 'Internship':
        search_query += ' internship'
    elif job_type == 'Job':
        search_query += ' full-time'
    if selection_type == 'Remote':
        search_query += ' remote'
    elif selection_type == 'Hybrid':
        search_query += ' hybrid'

    location_lower = location.lower()
    # Only map the location if it's in our mapping, otherwise use the provided location
    mapped_location = location  # Default to user-provided location
    if selection_type == 'Remote':
        # For remote jobs, still use a region but don't default to "Anywhere"
        mapped_location = next((country for key, country in COUNTRY_MAPPING.items() if key in location_lower), location)

    run_input = {
        "title": search_query,
        "location": mapped_location,
        "rows": max_jobs,
        "proxy": {
            "useApifyProxy": True,
        },
        "publishedAt": ""
    }

    try:
        print(f"Search Query: {search_query}")
        print(f"Original Location: {location}")
        print(f"Mapped Location: {mapped_location}")
        print(f"Run Input: {run_input}")
        print("Starting Apify LinkedIn Jobs Scraper (BHzefUZlZRKWxkTck)...")
        run = client.actor("BHzefUZlZRKWxkTck").call(run_input=run_input)
        
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        print(f"Raw dataset items count: {len(items)}")
        
        jobs = []
        for item in items:
            if len(jobs) >= max_jobs:
                break
            job = {
                'title': clean_text(item.get('title', 'N/A')),
                'company': clean_text(item.get('companyName', 'N/A')),
                'location': clean_text(item.get('location', mapped_location)),
                'url': item.get('jobUrl', ''),
                'description': clean_text(item.get('description', '')),
                'skills': [],
                'work_type': selection_type  # Set to Remote, Onsite, or Hybrid
            }
            jobs.append(job)
            print(f"Scraped job {len(jobs)}/{max_jobs}: {job['title']}")

        if not jobs:
            print("No jobs scraped. Check search query, location, Apify restrictions, or LinkedIn availability.")
        else:
            with open('jobs.json', 'w') as f:
                json.dump(jobs, f, indent=2)
            print(f"Successfully scraped {len(jobs)} jobs")
        return jobs

    except Exception as e:
        print(f"Apify scraping failed: {e}")
        return []

def process_jobs_batch(jobs: List[Dict[str, Any]], user_skills: List[str], user_experience: Dict[str, Any], client: Together) -> List[Dict[str, Any]]:
    for job in jobs:
        # Process each job
        job_experience = extract_job_experience(job['description'], client)
        job_skills = extract_job_skills(job['description'], client)
        
        # Calculate scores
        experience_score = calculate_experience_score(user_experience, job_experience, client)
        skill_score = match_skills(user_skills, job['description'], client)
        
        # Update job with results
        job['experience_score'] = experience_score
        job['skills'] = job_skills
        job['skill_score'] = skill_score
    
    return jobs

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    jobs = []
    user_skills = []
    user_experience = {"years_experience": 0, "job_roles": []}
    error_message = None

    try:
        client = init_together_client()
    except ValueError as e:
        error_message = f"Error: {e}"
        return render_template('index.html', jobs=jobs, user_skills=user_skills, user_experience=user_experience, error_message=error_message)

    if request.method == 'POST':
        role = request.form.get('role', '')
        location = request.form.get('location', '')
        job_type = request.form.get('job_type', '')
        selection_type = request.form.get('selection_type', 'Onsite')
        cv_file = request.files.get('cv')

        print(f"Form data received - Role: {role}, Location: {location}, Job Type: {job_type}, Work Type: {selection_type}")

        if not role or not location or not job_type:
            error_message = "Please provide role, location, and job type."
        elif cv_file:
            try:
                cv_text = parse_cv(cv_file)
                print(f"CV parsed successfully, extracting skills and experience...")
                # Process CV extraction
                user_skills = extract_skills(cv_text, client)
                user_experience = extract_experience(cv_text, client)
                print(f"Extracted skills: {user_skills}")
                print(f"Extracted experience: {user_experience}")
            except Exception as e:
                error_message = f"Failed to parse CV: {e}"
                print(f"CV parsing error: {e}")
        else:
            error_message = "Please upload a CV (PDF or text)."

        if not error_message:
            try:
                print(f"Starting LinkedIn job scraping for {role} in {location}...")
                scraped_jobs = scrape_linkedin_jobs(role, location, job_type, selection_type=selection_type)
                print(f"Scraping complete, found {len(scraped_jobs)} jobs")
                
                if not scraped_jobs:
                    error_message = (
                        "No jobs found. Possible causes: "
                        "1) Invalid search parameters (try 'software engineer', 'data analyst', etc.). "
                        "2) No jobs currently available for this role/location. "
                        "3) LinkedIn or Apify issues."
                    )
                    print(error_message)
                else:
                    print(f"Analyzing jobs for skill and experience match...")
                    # Process jobs
                    jobs = process_jobs_batch(scraped_jobs, user_skills, user_experience, client)
                    # Sort jobs by combined score
                    jobs = sorted(jobs, key=lambda x: (x['experience_score'] + x['skill_score']) / 2, reverse=True)
                    print(f"Job analysis complete, returning {len(jobs)} processed jobs")
            except Exception as e:
                error_message = f"Job scraping or matching failed: {e}"
                print(f"Job processing error: {e}")

    print(f"Rendering template with {len(jobs)} jobs, {len(user_skills)} skills, and error_message: {error_message}")
    return render_template('index.html',
                          jobs=jobs,
                          user_skills=user_skills,
                          user_experience=user_experience,
                          error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)