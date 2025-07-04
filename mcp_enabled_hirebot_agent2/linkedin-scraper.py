#!/usr/bin/env python3
import asyncio
import os
import re
import sys
from typing import List
from dotenv import load_dotenv
from apify_client import ApifyClient
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError as e:
    logger.error(f"MCP not available: {e}")
    MCP_AVAILABLE = False
    sys.exit(1)

from shared_models import Job, JobSearchRequest, JobSearchResponse

load_dotenv()

# Enhanced country mapping
COUNTRY_MAPPING = {
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "new york": "United States", 
    "california": "United States",
    "san francisco": "United States",
    "los angeles": "United States",
    "seattle": "United States",
    "chicago": "United States",
    "boston": "United States",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "london": "United Kingdom",
    "manchester": "United Kingdom",
    "birmingham": "United Kingdom",
    "canada": "Canada",
    "toronto": "Canada",
    "vancouver": "Canada",
    "montreal": "Canada",
    "india": "India",
    "bangalore": "India",
    "mumbai": "India",
    "delhi": "India",
    "hyderabad": "India",
    "pune": "India",
    "chennai": "India",
    "kolkata": "India",
    "ahmedabad": "India",
    "germany": "Germany",
    "berlin": "Germany",
    "munich": "Germany",
    "france": "France",
    "paris": "France",
    "australia": "Australia",
    "sydney": "Australia",
    "melbourne": "Australia",
}

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ''
    cleaned = re.sub(r'\s+', ' ', str(text).strip())
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    return cleaned

def extract_skills_from_description(description: str) -> List[str]:
    """Extract potential skills from job description"""
    if not description:
        return []
    
    skill_patterns = [
        r'\b(?:Python|Java|JavaScript|React|Node\.js|SQL|AWS|Docker|Kubernetes)\b',
        r'\b(?:Machine Learning|Data Science|AI|Analytics|Cloud|DevOps|TensorFlow|PyTorch)\b',
        r'\b(?:Project Management|Agile|Scrum|Leadership|Communication|Git|Linux)\b',
        r'\b(?:REST|API|Microservices|Database|MongoDB|PostgreSQL|MySQL|Redis)\b',
        r'\b(?:Angular|Vue|Express|Django|Flask|Spring|Hadoop|Spark|Kafka)\b'
    ]
    
    skills = []
    for pattern in skill_patterns:
        matches = re.findall(pattern, description, re.IGNORECASE)
        skills.extend(matches)
    
    return list(set(skills))

def map_location(location: str, selection_type: str) -> str:
    """Map location based on selection type and country mapping"""
    if not location:
        return "India"  # Default fallback
        
    location_lower = location.lower().strip()
    
    for key, country in COUNTRY_MAPPING.items():
        if key in location_lower:
            return country
    
    return location

def scrape_linkedin_jobs(search_request: JobSearchRequest) -> JobSearchResponse:
    """Scrape LinkedIn jobs using Apify"""
    logger.info(f"🔍 Starting job search for: {search_request.role} in {search_request.location}")
    
    api_token = os.getenv("APIFY_API_TOKEN")
    if not api_token:
        error_msg = "APIFY_API_TOKEN not set. Please set it in .env file."
        logger.error(error_msg)
        return JobSearchResponse(
            jobs=[],
            success=False,
            error_message=error_msg
        )
    
    try:
        client = ApifyClient(api_token)
        
        # Build search query
        search_query = search_request.role.strip()
        if search_request.job_type == 'Internship':
            search_query += ' internship'
        elif search_request.job_type == 'Job':
            search_query += ' full-time'
        
        if search_request.selection_type == 'Remote':
            search_query += ' remote'
        elif search_request.selection_type == 'Hybrid':
            search_query += ' hybrid'
        
        mapped_location = map_location(search_request.location, search_request.selection_type)
        
        # Multiple input formats to try
        input_formats = [
            # Format 1: Original format
            {
                "title": search_query,
                "location": mapped_location,
                "rows": min(search_request.max_jobs, 50),  # Limit to avoid timeouts
                "proxy": {"useApifyProxy": True},
                "publishedAt": ""
            },
            # Format 2: Alternative format
            {
                "queries": [search_query],
                "locations": [mapped_location],
                "maxResults": min(search_request.max_jobs, 50),
                "proxy": {"useApifyProxy": True},
                "timeFilter": "week"
            },
            # Format 3: Simple format
            {
                "keyword": search_query,
                "location": mapped_location,
                "count": min(search_request.max_jobs, 50),
                "proxy": {"useApifyProxy": True}
            }
        ]
        
        # Try different actor IDs with different input formats
        actor_configs = [
            ("BHzefUZlZRKWxkTck", input_formats[0]),  # Your original
            ("misceres/linkedin-jobs-scraper", input_formats[1]),
            ("curious_coder/linkedin-job-scraper", input_formats[2]),
        ]
        
        logger.info(f"Search Query: {search_query}")
        logger.info(f"Mapped Location: {mapped_location}")
        
        run = None
        last_error = None
        
        for actor_id, run_input in actor_configs:
            try:
                logger.info(f"Trying actor: {actor_id}")
                run = client.actor(actor_id).call(run_input=run_input)
                logger.info(f"✅ Successfully started run with actor: {actor_id}")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"❌ Actor {actor_id} failed: {e}")
                continue
        
        if not run:
            error_msg = f"All actor attempts failed. Last error: {last_error}"
            logger.error(error_msg)
            return JobSearchResponse(
                jobs=[],
                success=False,
                error_message=error_msg
            )
        
        # Get dataset items
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            error_msg = "No dataset ID returned from actor run"
            logger.error(error_msg)
            return JobSearchResponse(
                jobs=[],
                success=False,
                error_message=error_msg
            )
        
        items = list(client.dataset(dataset_id).iterate_items())
        logger.info(f"📊 Raw dataset items count: {len(items)}")
        
        if not items:
            error_msg = "No data returned from scraper. Try different search terms or check actor status."
            logger.warning(error_msg)
            return JobSearchResponse(
                jobs=[],
                success=False,
                error_message=error_msg
            )
        
        jobs = []
        for i, item in enumerate(items):
            if len(jobs) >= search_request.max_jobs:
                break
            
            try:
                # Handle different possible field names from different actors
                title = (item.get('title') or 
                        item.get('jobTitle') or 
                        item.get('position') or 
                        item.get('name') or 'N/A')
                
                company = (item.get('companyName') or 
                          item.get('company') or 
                          item.get('companyTitle') or 
                          item.get('employer') or 'N/A')
                
                location = (item.get('location') or 
                           item.get('jobLocation') or 
                           item.get('place') or 
                           mapped_location)
                
                url = (item.get('jobUrl') or 
                      item.get('link') or 
                      item.get('url') or 
                      item.get('applicationUrl') or '')
                
                description = (item.get('description') or 
                              item.get('jobDescription') or 
                              item.get('summary') or '')
                
                # Skip if essential fields are missing
                if title == 'N/A' or company == 'N/A':
                    logger.warning(f"Skipping item {i}: missing essential fields")
                    continue
                
                # Extract skills from description
                skills = extract_skills_from_description(description)
                    
                job = Job(
                    title=clean_text(title),
                    company=clean_text(company),
                    location=clean_text(location),
                    url=url,
                    description=clean_text(description)[:1000],  # Limit description length
                    skills=skills,
                    work_type=search_request.selection_type,
                    experience_score=0,
                    skill_score=0
                )
                jobs.append(job)
                logger.info(f"✅ Scraped job {len(jobs)}/{search_request.max_jobs}: {job.title} at {job.company}")
                
            except Exception as e:
                logger.warning(f"Error processing job item {i}: {e}")
                continue
        
        if not jobs:
            error_msg = "No valid jobs found in scraped data. Try different search parameters."
            logger.warning(error_msg)
            return JobSearchResponse(
                jobs=[],
                success=False,
                error_message=error_msg
            )
        
        logger.info(f"🎉 Successfully scraped {len(jobs)} jobs")
        return JobSearchResponse(jobs=jobs, success=True)
        
    except Exception as e:
        error_msg = f"Job scraping failed: {str(e)}"
        logger.error(error_msg)
        return JobSearchResponse(
            jobs=[],
            success=False,
            error_message=error_msg
        )

# Initialize MCP Server
server = Server("linkedin-scraper")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="scrape_jobs",
            description="Scrape LinkedIn jobs based on search criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string", 
                        "description": "Job role to search for (e.g., 'Software Engineer', 'Data Scientist', 'AI Engineer')"
                    },
                    "location": {
                        "type": "string", 
                        "description": "Location to search in (e.g., 'India', 'New York', 'Remote')"
                    },
                    "job_type": {
                        "type": "string", 
                        "enum": ["Job", "Internship"], 
                        "description": "Type of position"
                    },
                    "selection_type": {
                        "type": "string", 
                        "enum": ["Remote", "Onsite", "Hybrid"], 
                        "default": "Onsite",
                        "description": "Work arrangement type"
                    },
                    "max_jobs": {
                        "type": "integer", 
                        "default": 10, 
                        "minimum": 1, 
                        "maximum": 50, 
                        "description": "Maximum number of jobs to scrape"
                    }
                },
                "required": ["role", "location", "job_type"]
            }
        ),
        Tool(
            name="validate_location",
            description="Validate and map location string to standardized format",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location to validate"},
                    "selection_type": {"type": "string", "enum": ["Remote", "Onsite", "Hybrid"], "default": "Onsite"}
                },
                "required": ["location"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls"""
    logger.info(f"🔧 Tool called: {name} with arguments: {arguments}")
    
    if name == "scrape_jobs":
        try:
            # Validate required fields
            required_fields = ["role", "location", "job_type"]
            for field in required_fields:
                if not arguments.get(field):
                    raise ValueError(f"{field} is required")
            
            search_request = JobSearchRequest(**arguments)
            response = scrape_linkedin_jobs(search_request)
            
            result = response.model_dump()
            logger.info(f"📋 Returning {len(response.jobs)} jobs, success: {response.success}")
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            error_msg = f"Error processing scrape_jobs request: {str(e)}"
            logger.error(error_msg)
            error_response = JobSearchResponse(
                jobs=[],
                success=False,
                error_message=error_msg
            )
            return [TextContent(
                type="text",
                text=json.dumps(error_response.model_dump(), indent=2)
            )]
    
    elif name == "validate_location":
        try:
            location = arguments.get("location", "")
            if not location:
                raise ValueError("Location is required")
                
            selection_type = arguments.get("selection_type", "Onsite")
            mapped_location = map_location(location, selection_type)
            
            result = {
                "original_location": location,
                "mapped_location": mapped_location,
                "selection_type": selection_type,
                "is_mapped": mapped_location != location,
                "mapping_available": location.lower() in COUNTRY_MAPPING
            }
            
            logger.info(f"📍 Location validation: {location} -> {mapped_location}")
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            error_msg = f"Location validation failed: {str(e)}"
            logger.error(error_msg)
            return [TextContent(
                type="text", 
                text=json.dumps({"error": error_msg}, indent=2)
            )]
    
    else:
        error_msg = f"Unknown tool: {name}. Available tools: scrape_jobs, validate_location"
        logger.error(error_msg)
        return [TextContent(
            type="text",
            text=json.dumps({"error": error_msg}, indent=2)
        )]

async def main():
    """Run the LinkedIn scraper MCP server"""
    logger.info("🚀 Starting LinkedIn Job Scraper MCP Server...")
    
    # Check API token
    api_token = os.getenv("APIFY_API_TOKEN")
    if not api_token:
        logger.error("❌ APIFY_API_TOKEN not found in environment variables")
        logger.info("💡 Make sure you have a .env file with: APIFY_API_TOKEN=your_token_here")
        return
    else:
        logger.info("✅ APIFY_API_TOKEN found")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("📡 MCP Server started and listening...")
            await server.run(read_stream, write_stream, server.create_initialization_options())
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        raise

if __name__ == "__main__":
    if not MCP_AVAILABLE:
        print("❌ MCP not available. Please install it with:")
        print("pip install mcp")
        print("or")
        print("pip install model-context-protocol")
        sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        sys.exit(1)