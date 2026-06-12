# Project 3: AI Recommendation Logic

## Overview
A content-based filtering Tech Stack Recommender.
Enter your skills → get Top 3 career path recommendations.

## Algorithm Pipeline
1. **Ingestion** – User inputs 3+ skills
2. **TF-IDF Vectorization** – Skills → weighted numerical vectors
3. **Cosine Similarity** – User vector scored against 12 job roles
4. **Sort + Filter** – Returns Top-3 highest-scoring matches

## How to Run
```bash
python recommender.py
```

## Example
Your skills: python, machine learning, sql

Data Scientist       | Match: 0.8712
ML Engineer          | Match: 0.7934
Data Engineer        | Match: 0.6521


## Key Concepts
- Content-Based Filtering (no user history needed)
- TF-IDF weighting (penalizes generic skills)
- Cosine Similarity (angle-based, magnitude-invariant)
- Cold Start handled via onboarding input
