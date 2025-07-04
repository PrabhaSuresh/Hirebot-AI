import json
import base64
from typing import List, Dict, Any, Optional
from flask import Flask, request, render_template
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import os
import time

from shared_models import Job, UserExperience

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

class MCPClient:
    """MCP client wrapper for easier async handling"""
    
    def __init__(self):
        self.linkedin_session = None
        self.scoring_session = None
        self.linkedin_process = None
        self.scoring_process = None
    
    async def connect_linkedin_scraper(self):
        """Connect to LinkedIn scraper MCP server"""
        if self.linkedin_session is not None:
            return self.linkedin_session
            
        try:
            print("🔄 Connecting to LinkedIn scraper...")
            
            # Get absolute path to the script
            script_path = os.path.abspath("linkedin-scraper.py")
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"LinkedIn scraper script not found at {script_path}")
            
            server_params = StdioServerParameters(
                command=sys.executable,  # Use current Python interpreter
                args=[script_path],
                env=dict(os.environ)  # Pass current environment
            )
            
            # Create stdio client context manager
            stdio_context = stdio_client(server_params)
            read_stream, write_stream = await stdio_context.__aenter__()
            
            # Store the context for cleanup
            self.linkedin_stdio_context = stdio_context
            
            # Create ClientSession with the streams
            self.linkedin_session = ClientSession(read_stream, write_stream)
            await self.linkedin_session.__aenter__()
            
            # Initialize the session with timeout
            init_result = await asyncio.wait_for(
                self.linkedin_session.initialize(), 
                timeout=30.0
            )
            
            print("✅ LinkedIn scraper connected successfully")
            return self.linkedin_session
            
        except asyncio.TimeoutError:
            print("❌ Timeout connecting to LinkedIn scraper")
            await self._cleanup_linkedin_connection()
            raise Exception("Timeout connecting to LinkedIn scraper server")
        except Exception as e:
            print(f"❌ Failed to connect to LinkedIn scraper: {e}")
            await self._cleanup_linkedin_connection()
            raise Exception(f"Failed to connect to LinkedIn scraper: {str(e)}")
    
    async def connect_scoring_calculator(self):
        """Connect to scoring calculator MCP server"""
        if self.scoring_session is not None:
            return self.scoring_session
            
        try:
            print("🔄 Connecting to scoring calculator...")
            
            # Get absolute path to the script
            script_path = os.path.abspath("scoring-calculator.py")
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Scoring calculator script not found at {script_path}")
            
            server_params = StdioServerParameters(
                command=sys.executable,  # Use current Python interpreter
                args=[script_path],
                env=dict(os.environ)  # Pass current environment
            )
            
            # Create stdio client context manager
            stdio_context = stdio_client(server_params)
            read_stream, write_stream = await stdio_context.__aenter__()
            
            # Store the context for cleanup
            self.scoring_stdio_context = stdio_context
            
            # Create ClientSession with the streams
            self.scoring_session = ClientSession(read_stream, write_stream)
            await self.scoring_session.__aenter__()
            
            # Initialize the session with timeout
            init_result = await asyncio.wait_for(
                self.scoring_session.initialize(), 
                timeout=30.0
            )
            
            print("✅ Scoring calculator connected successfully")
            return self.scoring_session
            
        except asyncio.TimeoutError:
            print("❌ Timeout connecting to scoring calculator")
            await self._cleanup_scoring_connection()
            raise Exception("Timeout connecting to scoring calculator server")
        except Exception as e:
            print(f"❌ Failed to connect to scoring calculator: {e}")
            await self._cleanup_scoring_connection()
            raise Exception(f"Failed to connect to scoring calculator: {str(e)}")
    
    async def _cleanup_linkedin_connection(self):
        """Clean up LinkedIn connection"""
        try:
            if self.linkedin_session:
                await self.linkedin_session.__aexit__(None, None, None)
                self.linkedin_session = None
            if hasattr(self, 'linkedin_stdio_context'):
                await self.linkedin_stdio_context.__aexit__(None, None, None)
                delattr(self, 'linkedin_stdio_context')
        except Exception as e:
            print(f"⚠️  Error cleaning up LinkedIn connection: {e}")
    
    async def _cleanup_scoring_connection(self):
        """Clean up scoring connection"""
        try:
            if self.scoring_session:
                await self.scoring_session.__aexit__(None, None, None)
                self.scoring_session = None
            if hasattr(self, 'scoring_stdio_context'):
                await self.scoring_stdio_context.__aexit__(None, None, None)
                delattr(self, 'scoring_stdio_context')
        except Exception as e:
            print(f"⚠️  Error cleaning up scoring connection: {e}")
    
    async def scrape_jobs(self, role: str, location: str, job_type: str, selection_type: str = "Onsite", max_jobs: int = 10) -> Dict:
        """Scrape jobs using LinkedIn scraper server"""
        try:
            session = await self.connect_linkedin_scraper()
            
            print(f"🔍 Scraping jobs for {role} in {location}...")
            
            # Call with timeout
            result = await asyncio.wait_for(
                session.call_tool(
                    "scrape_jobs",
                    {
                        "role": role,
                        "location": location,
                        "job_type": job_type,
                        "selection_type": selection_type,
                        "max_jobs": max_jobs
                    }
                ),
                timeout=120.0  # 2 minutes timeout for job scraping
            )
            
            # Handle different response formats
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        response_text = content.text
                    else:
                        response_text = str(content)
                else:
                    response_text = str(result.content)
            else:
                response_text = str(result)
            
            return json.loads(response_text)
            
        except asyncio.TimeoutError:
            print("❌ Timeout during job scraping")
            return {
                'success': False,
                'error_message': 'Job scraping timed out. Please try again with fewer jobs or different criteria.',
                'jobs': []
            }
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in scrape_jobs: {e}")
            print(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
            return {
                'success': False,
                'error_message': f'Invalid response from LinkedIn scraper: {str(e)}',
                'jobs': []
            }
        except Exception as e:
            print(f"❌ Error in scrape_jobs: {e}")
            # Reset connection on error
            await self._cleanup_linkedin_connection()
            return {
                'success': False,
                'error_message': f'LinkedIn scraper error: {str(e)}',
                'jobs': []
            }
    
    async def analyze_cv(self, cv_text: str) -> Dict:
        """Analyze CV using scoring calculator server"""
        try:
            session = await self.connect_scoring_calculator()
            
            print("📄 Analyzing CV...")
            
            # Call with timeout
            result = await asyncio.wait_for(
                session.call_tool(
                    "analyze_cv",
                    {"cv_text": cv_text}
                ),
                timeout=60.0  # 1 minute timeout for CV analysis
            )
            
            # Handle different response formats
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        response_text = content.text
                    else:
                        response_text = str(content)
                else:
                    response_text = str(result.content)
            else:
                response_text = str(result)
            
            return json.loads(response_text)
            
        except asyncio.TimeoutError:
            print("❌ Timeout during CV analysis")
            return {
                'success': False,
                'error_message': 'CV analysis timed out. Please try again.',
                'skills': [],
                'experience': {'years_experience': 0, 'job_roles': []}
            }
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in analyze_cv: {e}")
            print(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
            return {
                'success': False,
                'error_message': f'Invalid response from CV analyzer: {str(e)}',
                'skills': [],
                'experience': {'years_experience': 0, 'job_roles': []}
            }
        except Exception as e:
            print(f"❌ Error in analyze_cv: {e}")
            # Reset connection on error
            await self._cleanup_scoring_connection()
            return {
                'success': False,
                'error_message': f'CV analysis error: {str(e)}',
                'skills': [],
                'experience': {'years_experience': 0, 'job_roles': []}
            }
    
    async def match_jobs(self, jobs: List[Dict], user_skills: List[str], user_experience: Dict) -> Dict:
        """Match jobs using scoring calculator server"""
        try:
            session = await self.connect_scoring_calculator()
            
            print("🎯 Matching jobs with user profile...")
            
            # Call with timeout
            result = await asyncio.wait_for(
                session.call_tool(
                    "match_jobs",
                    {
                        "jobs": jobs,
                        "user_skills": user_skills,
                        "user_experience": user_experience
                    }
                ),
                timeout=90.0  # 1.5 minutes timeout for job matching
            )
            
            # Handle different response formats
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        response_text = content.text
                    else:
                        response_text = str(content)
                else:
                    response_text = str(result.content)
            else:
                response_text = str(result)
            
            return json.loads(response_text)
            
        except asyncio.TimeoutError:
            print("❌ Timeout during job matching")
            return {
                'success': False,
                'error_message': 'Job matching timed out. Please try again.',
                'scored_jobs': []
            }
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in match_jobs: {e}")
            print(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
            return {
                'success': False,
                'error_message': f'Invalid response from job matcher: {str(e)}',
                'scored_jobs': []
            }
        except Exception as e:
            print(f"❌ Error in match_jobs: {e}")
            # Reset connection on error
            await self._cleanup_scoring_connection()
            return {
                'success': False,
                'error_message': f'Job matching error: {str(e)}',
                'scored_jobs': []
            }
    
    async def close(self):
        """Close MCP connections"""
        print("🔌 Closing MCP connections...")
        await self._cleanup_linkedin_connection()
        await self._cleanup_scoring_connection()
        print("✅ MCP connections closed")

# Global MCP client instance
mcp_client = MCPClient()

def run_async(coro):
    """Run async function in thread pool with proper cleanup"""
    def run_in_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        except Exception as e:
            print(f"❌ Error in async execution: {e}")
            raise
        finally:
            # Clean up pending tasks
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as cleanup_error:
                print(f"⚠️  Error during cleanup: {cleanup_error}")
            finally:
                loop.close()
    
    return run_in_loop()

def parse_cv(file):
    """Parse CV file content"""
    try:
        if file.filename.endswith('.pdf'):
            import pdfplumber
            from io import BytesIO
            
            text = ''
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
            return text.strip()
        else:  # Assume text file
            return file.read().decode('utf-8')
    except Exception as e:
        print(f"❌ Error parsing CV: {e}")
        raise Exception(f"Failed to parse CV file: {str(e)}")

async def process_job_matching(role: str, location: str, job_type: str, selection_type: str, cv_text: str, max_jobs: int = 10):
    """Main async function to process job matching"""
    try:
        print(f"🚀 Starting job matching process for {role} in {location}")
        
        # Step 1: Scrape jobs
        print("📋 Step 1: Scraping jobs...")
        jobs_response = await mcp_client.scrape_jobs(role, location, job_type, selection_type, max_jobs)
        
        if not jobs_response.get('success', False):
            error_msg = jobs_response.get('error_message', 'Unknown error')
            print(f"❌ Job scraping failed: {error_msg}")
            return {
                'success': False,
                'error_message': error_msg,
                'jobs': [],
                'user_skills': [],
                'user_experience': {'years_experience': 0, 'job_roles': []}
            }
        
        jobs = jobs_response.get('jobs', [])
        print(f"✅ Found {len(jobs)} jobs")
        
        if not jobs:
            return {
                'success': False,
                'error_message': 'No jobs found for the given criteria. Try different search terms or location.',
                'jobs': [],
                'user_skills': [],
                'user_experience': {'years_experience': 0, 'job_roles': []}
            }
        
        # Step 2: Analyze CV
        print("🔍 Step 2: Analyzing CV...")
        cv_response = await mcp_client.analyze_cv(cv_text)
        
        if not cv_response.get('success', False):
            error_msg = cv_response.get('error_message', 'Unknown error')
            print(f"❌ CV analysis failed: {error_msg}")
            return {
                'success': False,
                'error_message': error_msg,
                'jobs': [],
                'user_skills': [],
                'user_experience': {'years_experience': 0, 'job_roles': []}
            }
        
        user_skills = cv_response.get('skills', [])
        user_experience = cv_response.get('experience', {'years_experience': 0, 'job_roles': []})
        print(f"✅ Extracted {len(user_skills)} skills and {user_experience.get('years_experience', 0)} years experience")
        
        # Step 3: Match jobs
        print("🎯 Step 3: Matching jobs...")
        matching_response = await mcp_client.match_jobs(jobs, user_skills, user_experience)
        
        if not matching_response.get('success', False):
            error_msg = matching_response.get('error_message', 'Unknown error')
            print(f"❌ Job matching failed: {error_msg}")
            return {
                'success': False,
                'error_message': error_msg,
                'jobs': [],
                'user_skills': user_skills,
                'user_experience': user_experience
            }
        
        scored_jobs = matching_response.get('scored_jobs', [])
        print(f"🎉 Successfully matched {len(scored_jobs)} jobs")
        
        return {
            'success': True,
            'jobs': scored_jobs,
            'user_skills': user_skills,
            'user_experience': user_experience
        }
        
    except Exception as e:
        print(f"💥 Error in job matching process: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error_message': f'Processing failed: {str(e)}',
            'jobs': [],
            'user_skills': [],
            'user_experience': {'years_experience': 0, 'job_roles': []}
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for the job matching application"""
    jobs = []
    user_skills = []
    user_experience = {"years_experience": 0, "job_roles": []}
    error_message = None

    if request.method == 'POST':
        role = request.form.get('role', '').strip()
        location = request.form.get('location', '').strip()
        job_type = request.form.get('job_type', '').strip()
        selection_type = request.form.get('selection_type', 'Onsite')
        cv_file = request.files.get('cv')

        print(f"📝 Form data received - Role: {role}, Location: {location}, Job Type: {job_type}, Work Type: {selection_type}")

        # Validate inputs
        if not role or not location or not job_type:
            error_message = "Please provide role, location, and job type."
            print(f"❌ Validation failed: Missing required fields")
        elif not cv_file or cv_file.filename == '':
            error_message = "Please upload a CV (PDF or text)."
            print(f"❌ Validation failed: No CV file uploaded")
        else:
            try:
                # Parse CV
                print("📄 Parsing CV file...")
                cv_text = parse_cv(cv_file)
                print(f"✅ CV parsed successfully, length: {len(cv_text)} characters")
                
                # Process job matching asynchronously
                print("🔄 Starting async job matching process...")
                result = run_async(process_job_matching(role, location, job_type, selection_type, cv_text))
                
                if result['success']:
                    jobs = result['jobs']
                    user_skills = result['user_skills']
                    user_experience = result['user_experience']
                    print(f"🎊 Successfully processed {len(jobs)} jobs")
                else:
                    error_message = result['error_message']
                    user_skills = result['user_skills']
                    user_experience = result['user_experience']
                    print(f"❌ Job matching failed: {error_message}")
                    
            except Exception as e:
                error_message = f"Processing failed: {str(e)}"
                print(f"💥 Error in request processing: {e}")
                import traceback
                traceback.print_exc()

    print(f"🎨 Rendering template with {len(jobs)} jobs, {len(user_skills)} skills, error: {error_message}")
    return render_template('index.html',
                          jobs=jobs,
                          user_skills=user_skills,
                          user_experience=user_experience,
                          error_message=error_message)

@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "job-matcher-client"}

# Graceful shutdown handler
import atexit

def cleanup_on_exit():
    """Clean up MCP connections on exit"""
    try:
        asyncio.run(mcp_client.close())
    except:
        pass

atexit.register(cleanup_on_exit)

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 Starting Job Matcher Client Application...")
    print("=" * 60)
    print("🔗 This client connects to:")
    print("   📊 LinkedIn Scraper MCP Server (linkedin-scraper.py)")
    print("   🎯 Scoring Calculator MCP Server (scoring-calculator.py)")
    print("=" * 60)
    
    # Check if MCP server files exist
    required_files = ["linkedin-scraper.py", "scoring-calculator.py", "shared_models.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("   Please ensure all MCP server files are in the same directory.")
        sys.exit(1)
    else:
        print("✅ All required MCP server files found")
    
    print("🌐 Starting Flask application on http://127.0.0.1:5000")
    print("=" * 60)
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        cleanup_on_exit()
    except Exception as e:
        print(f"💥 Application error: {e}")
    finally:
        print("🔌 Application stopped")