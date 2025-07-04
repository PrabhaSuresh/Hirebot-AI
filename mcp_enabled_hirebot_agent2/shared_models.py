from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Job(BaseModel):
    title: str
    company: str
    location: str
    url: str
    description: str
    skills: List[str] = []
    work_type: str = "Onsite"  # Remote, Onsite, Hybrid
    experience_score: int = 0
    skill_score: int = 0

    @property
    def combined_score(self) -> float:
        return (self.experience_score + self.skill_score) / 2

class UserExperience(BaseModel):
    years_experience: int = 0
    job_roles: List[str] = []

class JobExperience(BaseModel):
    required_years: int = 0
    preferred_roles: List[str] = []

class JobSearchRequest(BaseModel):
    role: str
    location: str
    job_type: str  # Job, Internship
    selection_type: str = "Onsite"  # Remote, Onsite, Hybrid
    max_jobs: int = 10

class JobSearchResponse(BaseModel):
    jobs: List[Job]
    success: bool = True
    error_message: Optional[str] = None

class CVAnalysisRequest(BaseModel):
    cv_text: str

class CVAnalysisResponse(BaseModel):
    skills: List[str]
    experience: UserExperience
    success: bool = True
    error_message: Optional[str] = None

class JobMatchingRequest(BaseModel):
    jobs: List[Job]
    user_skills: List[str]
    user_experience: UserExperience

class JobMatchingResponse(BaseModel):
    scored_jobs: List[Job]
    success: bool = True
    error_message: Optional[str] = None