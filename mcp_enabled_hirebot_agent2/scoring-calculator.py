import asyncio
import os
import json
import re
from typing import List, Dict, Any, Optional
from functools import lru_cache
from dotenv import load_dotenv
from together import Together
import pdfplumber
from io import BytesIO
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import concurrent.futures
import time

from shared_models import (
    Job, UserExperience, JobExperience, CVAnalysisRequest, CVAnalysisResponse,
    JobMatchingRequest, JobMatchingResponse
)

load_dotenv()

# Configuration
MAX_CONCURRENT_JOBS = 5  # Limit concurrent job processing
REQUEST_TIMEOUT = 30  # Timeout per AI request in seconds
BATCH_SIZE = 10  # Process jobs in batches
TOP_JOBS_LIMIT = 10  # Return only top N job matches

# Enhanced skill normalization mapping
SKILL_SYNONYMS = {
    "javascript": ["js", "javascript", "java script", "ecmascript"],
    "python": ["python", "python programming", "python development"],
    "react": ["react", "react.js", "reactjs", "react js"],
    "machine learning": ["ml", "machine learning", "artificial intelligence", "ai"],
    "sql": ["sql", "structured query language", "mysql", "postgresql"],
    "node.js": ["nodejs", "node js", "node.js", "node"],
    "angular": ["angular", "angularjs", "angular.js"],
    "vue": ["vue", "vue.js", "vuejs", "vue js"],
    "aws": ["aws", "amazon web services", "amazon aws"],
    "docker": ["docker", "containerization", "containers"],
    "kubernetes": ["k8s", "kubernetes", "k8", "kube"],
    "git": ["git", "version control", "github", "gitlab"],
    "css": ["css", "css3", "cascading style sheets"],
    "html": ["html", "html5", "hypertext markup language"],
    "mongodb": ["mongodb", "mongo", "mongo db"],
    "postgresql": ["postgresql", "postgres", "psql"],
    "typescript": ["typescript", "ts"],
    "java": ["java", "java programming"],
    "c++": ["c++", "cpp", "c plus plus"],
    "c#": ["c#", "csharp", "c sharp"],
    "golang": ["go", "golang", "go lang"],
    "ruby": ["ruby", "ruby programming"],
    "php": ["php", "php programming"],
    "swift": ["swift", "swift programming"],
    "kotlin": ["kotlin", "kotlin programming"],
    "scala": ["scala", "scala programming"],
    "rust": ["rust", "rust programming"],
    "tensorflow": ["tensorflow", "tf", "tensor flow"],
    "pytorch": ["pytorch", "torch", "py torch"],
    "scikit-learn": ["scikit-learn", "sklearn", "sci-kit learn"],
    "pandas": ["pandas", "python pandas"],
    "numpy": ["numpy", "np", "num py"],
    "flask": ["flask", "python flask"],
    "django": ["django", "python django"],
    "spring": ["spring", "spring boot", "spring framework"],
    "express": ["express", "express.js", "expressjs"],
    "redis": ["redis", "redis db"],
    "elasticsearch": ["elasticsearch", "elastic search"],
    "apache": ["apache", "apache server"],
    "nginx": ["nginx", "nginx server"],
    "jenkins": ["jenkins", "ci/cd", "continuous integration"],
    "tableau": ["tableau", "tableau desktop"],
    "power bi": ["power bi", "powerbi", "microsoft power bi"],
    "excel": ["excel", "microsoft excel", "ms excel"],
    "jira": ["jira", "atlassian jira"],
    "agile": ["agile", "agile methodology", "scrum"],
    "devops": ["devops", "dev ops", "development operations"],
    "api": ["api", "rest api", "restful api"],
    "microservices": ["microservices", "micro services"],
    "blockchain": ["blockchain", "block chain"],
    "cyber security": ["cybersecurity", "cyber security", "information security"],
    "cloud computing": ["cloud", "cloud computing", "cloud services"],
    "data science": ["data science", "data analytics", "data analysis"],
    "business intelligence": ["bi", "business intelligence", "business analytics"],
    "project management": ["project management", "pm", "project coordination"],
    "leadership": ["leadership", "team leadership", "team management"],
    "communication": ["communication", "verbal communication", "written communication"],
    "teamwork": ["teamwork", "collaboration", "team collaboration"],
    "problem solving": ["problem solving", "analytical thinking", "critical thinking"]
}

def normalize_skill(skill: str) -> str:
    """Normalize skill to standard form"""
    skill_lower = skill.lower().strip()
    
    # Remove common suffixes/prefixes
    skill_lower = re.sub(r'\b(programming|development|coding|language|framework|library|tool|software)\b', '', skill_lower).strip()
    
    # Remove extra spaces
    skill_lower = re.sub(r'\s+', ' ', skill_lower).strip()
    
    # Find the canonical form
    for canonical, synonyms in SKILL_SYNONYMS.items():
        if skill_lower in synonyms:
            return canonical
    
    return skill_lower

def clean_text(text):
    """Clean and normalize text"""
    return re.sub(r'\s+', ' ', text.strip()) if text else ''

def init_together_client():
    """Initialize Together AI client with timeout settings"""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set. Please set it in .env or environment.")
    
    client = Together(api_key=api_key)
    # Set default timeout for requests
    client.timeout = REQUEST_TIMEOUT
    return client

def parse_cv_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Parse CV from bytes (for PDF or text files)"""
    text = ''
    try:
        if filename.endswith('.pdf'):
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        else:  # Assume text file
            text = file_bytes.decode('utf-8')
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        raise
    
    return clean_text(text)

async def make_ai_request_with_timeout(client: Together, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Optional[str]:
    """Make AI request with timeout and error handling"""
    try:
        # Run the AI request in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                client.chat.completions.create,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Wait for completion with timeout
            response = await asyncio.wait_for(
                loop.run_in_executor(None, future.result),
                timeout=REQUEST_TIMEOUT
            )
            
            return response.choices[0].message.content.strip()
    
    except asyncio.TimeoutError:
        print(f"AI request timed out after {REQUEST_TIMEOUT} seconds")
        return None
    except Exception as e:
        print(f"AI request failed: {e}")
        return None

async def extract_skills_async_enhanced(cv_text: str, client: Together) -> List[str]:
    """Enhanced skill extraction with better prompts"""
    max_cv_length = 2000
    if len(cv_text) > max_cv_length:
        cv_text = cv_text[:max_cv_length] + "..."
    
    prompt = f"""
Extract professional skills from the CV. Use standard, canonical names for technologies and skills.

Rules:
- Use "JavaScript" not "JS"
- Use "Python" not "Python programming"
- Use "React" not "React.js" or "ReactJS"
- Use "Machine Learning" not "ML"
- Use "SQL" not "MySQL" or "PostgreSQL" (unless specific database mentioned)
- Use "AWS" not "Amazon Web Services"
- Include both technical and soft skills
- Be concise and specific

Return only the skills as a comma-separated list.

CV Text:
{cv_text}
"""
    
    response_text = await make_ai_request_with_timeout(client, prompt, max_tokens=150)
    if not response_text:
        return []
    
    try:
        skills = [skill.strip() for skill in response_text.split(',') if skill.strip()]
        # Additional normalization
        normalized_skills = [normalize_skill(skill) for skill in skills]
        return list(set(normalized_skills))[:20]  # Remove duplicates and limit
    except:
        return []

async def extract_skills_async(cv_text: str, client: Together) -> List[str]:
    """Extract skills from CV text using AI with async handling - kept for backward compatibility"""
    return await extract_skills_async_enhanced(cv_text, client)

async def extract_experience_async(cv_text: str, client: Together) -> UserExperience:
    """Extract experience information from CV text using AI with async handling"""
    # Truncate CV text if too long
    max_cv_length = 2000
    if len(cv_text) > max_cv_length:
        cv_text = cv_text[:max_cv_length] + "..."
    
    prompt = f"""
Analyze the following CV text and extract the total years of professional experience and job roles. Return as JSON:
{{"years_experience": <integer>, "job_roles": ["role1", "role2"]}}

CV Text:
{cv_text}
"""
    
    response_text = await make_ai_request_with_timeout(client, prompt, max_tokens=200)
    if not response_text:
        return UserExperience(years_experience=0, job_roles=[])
    
    try:
        # Clean the response to extract JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return UserExperience(
                years_experience=max(0, int(result.get("years_experience", 0))),
                job_roles=result.get("job_roles", [])[:10]  # Limit roles
            )
    except Exception as e:
        print(f"Error parsing experience: {e}")
    
    return UserExperience(years_experience=0, job_roles=[])

@lru_cache(maxsize=50)  # Reduced cache size
def extract_job_skills_cached(job_description: str) -> str:
    """Cached version of job skills extraction (synchronous for caching)"""
    return job_description[:1000]  # Truncate for processing

async def extract_job_skills_async_enhanced(job_description: str, client: Together) -> List[str]:
    """Enhanced job skills extraction with better prompts"""
    # Use cached truncated description
    truncated_desc = extract_job_skills_cached(job_description)
    
    prompt = f"""
Extract required skills from this job description. Use standard, canonical names for technologies and skills.

Rules:
- Use "JavaScript" not "JS"
- Use "Python" not "Python programming"
- Use "React" not "React.js" or "ReactJS"
- Use "Machine Learning" not "ML"
- Use "SQL" not "MySQL" or "PostgreSQL" (unless specific database mentioned)
- Use "AWS" not "Amazon Web Services"
- Include both technical and soft skills
- Be concise and specific

Return as comma-separated list:

{truncated_desc}
"""
    
    response_text = await make_ai_request_with_timeout(client, prompt, max_tokens=150)
    if not response_text:
        return []
    
    try:
        skills = [skill.strip() for skill in response_text.split(',') if skill.strip()]
        # Additional normalization
        normalized_skills = [normalize_skill(skill) for skill in skills]
        return list(set(normalized_skills))[:15]  # Remove duplicates and limit
    except:
        return []

async def extract_job_skills_async(job_description: str, client: Together) -> List[str]:
    """Extract required skills from job description using AI with async handling - kept for backward compatibility"""
    return await extract_job_skills_async_enhanced(job_description, client)

def simple_fuzzy_match(str1: str, str2: str, threshold: float = 0.75) -> bool:
    """Simple fuzzy string matching without external dependencies"""
    if not str1 or not str2:
        return False
    
    # Convert to lowercase and remove extra spaces
    str1 = re.sub(r'\s+', ' ', str1.lower().strip())
    str2 = re.sub(r'\s+', ' ', str2.lower().strip())
    
    # Exact match
    if str1 == str2:
        return True
    
    # Check if one is contained in the other
    if str1 in str2 or str2 in str1:
        return True
    
    # Simple character-based similarity
    longer = str1 if len(str1) > len(str2) else str2
    shorter = str2 if len(str1) > len(str2) else str1
    
    if len(longer) == 0:
        return True
    
    # Calculate similarity based on character overlap
    matches = 0
    for char in shorter:
        if char in longer:
            matches += 1
    
    similarity = matches / len(longer)
    return similarity >= threshold

def calculate_skill_overlap_score_enhanced(user_skills: List[str], job_skills: List[str]) -> int:
    """Enhanced skill matching with normalization and simple fuzzy matching"""
    if not user_skills or not job_skills:
        return 0
    
    # Step 1: Normalize skills
    user_skills_normalized = [normalize_skill(skill) for skill in user_skills]
    job_skills_normalized = [normalize_skill(skill) for skill in job_skills]
    
    # Step 2: Exact matches first
    exact_matches = set(user_skills_normalized) & set(job_skills_normalized)
    matched_count = len(exact_matches)
    
    # Step 3: Fuzzy matching for remaining job skills
    remaining_job_skills = [skill for skill in job_skills_normalized if skill not in exact_matches]
    remaining_user_skills = [skill for skill in user_skills_normalized if skill not in exact_matches]
    
    for job_skill in remaining_job_skills:
        for user_skill in remaining_user_skills:
            if simple_fuzzy_match(job_skill, user_skill, threshold=0.75):
                matched_count += 1
                break  # Only count each job skill once
    
    score = (matched_count / len(job_skills_normalized)) * 100
    return min(100, int(score))

def calculate_skill_overlap_score(user_skills: List[str], job_skills: List[str]) -> int:
    """Calculate skill matching score using enhanced method"""
    return calculate_skill_overlap_score_enhanced(user_skills, job_skills)

async def calculate_scores_batch(user_skills: List[str], user_experience: UserExperience, 
                               jobs_batch: List[Job], client: Together) -> List[Job]:
    """Calculate scores for a batch of jobs concurrently"""
    tasks = []
    
    for job in jobs_batch:
        task = calculate_job_scores(user_skills, user_experience, job, client)
        tasks.append(task)
    
    # Process batch concurrently with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    
    async def process_with_semaphore(task):
        async with semaphore:
            return await task
    
    results = await asyncio.gather(*[process_with_semaphore(task) for task in tasks], 
                                 return_exceptions=True)
    
    # Filter out failed jobs
    processed_jobs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Job processing failed for {jobs_batch[i].title}: {result}")
            # Add job with zero scores if processing fails
            job = jobs_batch[i]
            job.experience_score = 0
            job.skill_score = 0
            job.skills = []
            processed_jobs.append(job)
        else:
            processed_jobs.append(result)
    
    return processed_jobs

async def calculate_job_scores(user_skills: List[str], user_experience: UserExperience, 
                             job: Job, client: Together) -> Job:
    """Calculate scores for a single job"""
    try:
        # Extract job skills
        job_skills = await extract_job_skills_async(job.description, client)
        
        # Calculate skill overlap score (enhanced method)
        skill_score = calculate_skill_overlap_score(user_skills, job_skills)
        
        # Calculate experience score (simple heuristic)
        experience_score = calculate_experience_heuristic_score(user_experience, job.description)
        
        # Update job with scores
        job.experience_score = experience_score
        job.skill_score = skill_score
        job.skills = job_skills
        
        return job
        
    except Exception as e:
        print(f"Error calculating scores for job {job.title}: {e}")
        job.experience_score = 0
        job.skill_score = 0
        job.skills = []
        return job

def calculate_experience_heuristic_score(user_experience: UserExperience, job_description: str) -> int:
    """Calculate experience score using heuristics to avoid AI calls"""
    # Simple heuristic based on keywords in job description
    job_desc_lower = job_description.lower()
    
    # Look for experience requirements
    experience_score = 50  # Base score
    
    # Check for role matches
    user_roles_lower = [role.lower() for role in user_experience.job_roles]
    for role in user_roles_lower:
        if role in job_desc_lower:
            experience_score += 20
            break
    
    # Adjust based on years of experience
    if user_experience.years_experience >= 5:
        experience_score += 20
    elif user_experience.years_experience >= 2:
        experience_score += 10
    
    return min(100, experience_score)

async def analyze_cv_async(cv_text: str) -> CVAnalysisResponse:
    """Analyze CV and extract skills and experience with async processing"""
    try:
        client = init_together_client()
        
        print("Extracting skills and experience from CV...")
        
        # Run both extractions concurrently
        skills_task = extract_skills_async(cv_text, client)
        experience_task = extract_experience_async(cv_text, client)
        
        skills, experience = await asyncio.gather(skills_task, experience_task)
        
        return CVAnalysisResponse(
            skills=skills,
            experience=experience,
            success=True
        )
    except Exception as e:
        print(f"CV analysis failed: {e}")
        return CVAnalysisResponse(
            skills=[],
            experience=UserExperience(),
            success=False,
            error_message=str(e)
        )

async def match_jobs_async(request: JobMatchingRequest) -> JobMatchingResponse:
    """Match jobs with user profile using async processing and batching"""
    try:
        client = init_together_client()
        
        print(f"Processing {len(request.jobs)} jobs in batches of {BATCH_SIZE} (returning top {TOP_JOBS_LIMIT})")
        
        all_scored_jobs = []
        
        # Process jobs in batches to avoid overwhelming the API
        for i in range(0, len(request.jobs), BATCH_SIZE):
            batch = request.jobs[i:i + BATCH_SIZE]
            print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(request.jobs) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            try:
                # Process batch with timeout
                scored_batch = await asyncio.wait_for(
                    calculate_scores_batch(request.user_skills, request.user_experience, batch, client),
                    timeout=60  # 60 seconds per batch
                )
                all_scored_jobs.extend(scored_batch)
                
            except asyncio.TimeoutError:
                print(f"Batch {i//BATCH_SIZE + 1} timed out, adding jobs with zero scores")
                for job in batch:
                    job.experience_score = 0
                    job.skill_score = 0
                    job.skills = []
                    all_scored_jobs.append(job)
            
            # Early termination optimization: if we have enough high-scoring jobs, we can stop processing
            if len(all_scored_jobs) >= TOP_JOBS_LIMIT * 2:  # Process 2x to ensure we get the best matches
                print(f"Early termination: processed {len(all_scored_jobs)} jobs, sorting to find top matches")
                break
            
            # Small delay between batches to avoid rate limiting
            if i + BATCH_SIZE < len(request.jobs):
                await asyncio.sleep(1)
        
        # Sort by combined score and return only top N
        all_scored_jobs.sort(key=lambda x: x.combined_score, reverse=True)
        top_jobs = all_scored_jobs[:TOP_JOBS_LIMIT]
        
        print(f"Returning top {len(top_jobs)} job matches out of {len(all_scored_jobs)} processed")
        
        return JobMatchingResponse(
            scored_jobs=top_jobs,
            success=True
        )
        
    except Exception as e:
        print(f"Job matching failed: {e}")
        return JobMatchingResponse(
            scored_jobs=[],
            success=False,
            error_message=str(e)
        )

# Initialize MCP Server
server = Server("scoring-calculator")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="parse_cv",
            description="Parse CV text from file bytes",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_bytes": {"type": "string", "description": "Base64 encoded file bytes"},
                    "filename": {"type": "string", "description": "Original filename"}
                },
                "required": ["file_bytes", "filename"]
            }
        ),
        Tool(
            name="analyze_cv",
            description="Extract skills and experience from CV text",
            inputSchema={
                "type": "object",
                "properties": {
                    "cv_text": {"type": "string", "description": "CV text content"}
                },
                "required": ["cv_text"]
            }
        ),
        Tool(
            name="match_jobs",
            description="Calculate matching scores for jobs based on user profile",
            inputSchema={
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "company": {"type": "string"},
                                "location": {"type": "string"},
                                "url": {"type": "string"},
                                "description": {"type": "string"},
                                "work_type": {"type": "string"}
                            }
                        }
                    },
                    "user_skills": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "user_experience": {
                        "type": "object",
                        "properties": {
                            "years_experience": {"type": "integer"},
                            "job_roles": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["jobs", "user_skills", "user_experience"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls with proper async support"""
    if name == "parse_cv":
        try:
            import base64
            file_bytes = base64.b64decode(arguments["file_bytes"])
            filename = arguments["filename"]
            cv_text = parse_cv_from_bytes(file_bytes, filename)
            
            return [TextContent(
                type="text",
                text=json.dumps({"cv_text": cv_text, "success": True}, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"cv_text": "", "success": False, "error": str(e)}, indent=2)
            )]
    
    elif name == "analyze_cv":
        try:
            cv_text = arguments["cv_text"]
            response = await analyze_cv_async(cv_text)
            
            return [TextContent(
                type="text",
                text=json.dumps(response.model_dump(), indent=2)
            )]
        except Exception as e:
            error_response = CVAnalysisResponse(
                skills=[],
                experience=UserExperience(),
                success=False,
                error_message=str(e)
            )
            return [TextContent(
                type="text",
                text=json.dumps(error_response.model_dump(), indent=2)
            )]
    
    elif name == "match_jobs":
        try:
            # Convert dict to proper models
            jobs = [Job(**job_data) for job_data in arguments["jobs"]]
            user_experience = UserExperience(**arguments["user_experience"])
            request = JobMatchingRequest(
                jobs=jobs,
                user_skills=arguments["user_skills"],
                user_experience=user_experience
            )
            
            response = await match_jobs_async(request)
            
            return [TextContent(
                type="text",
                text=json.dumps(response.model_dump(), indent=2)
            )]
        except Exception as e:
            error_response = JobMatchingResponse(
                scored_jobs=[],
                success=False,
                error_message=str(e)
            )
            return [TextContent(
                type="text",
                text=json.dumps(error_response.model_dump(), indent=2)
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2)
        )]

async def main():
    """Run the scoring calculator MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    print("Starting Enhanced Scoring Calculator MCP Server...")
    asyncio.run(main())