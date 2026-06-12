# Project 3: AI Recommendation Logic - Tech Stack Recommender
# DecodeLabs Internship | Batch 2026
# Algorithm: Content-Based Filtering using TF-IDF + Cosine Similarity

import math
from collections import defaultdict

# ─────────────────────────────────────────────
# DATASET: Job Roles with required skills
# ─────────────────────────────────────────────
job_roles = [
    {"title": "Data Scientist",       "skills": ["python", "machine learning", "sql", "statistics", "data analysis", "pandas", "numpy"]},
    {"title": "Machine Learning Engineer", "skills": ["python", "machine learning", "tensorflow", "deep learning", "algorithms", "numpy"]},
    {"title": "Backend Developer",    "skills": ["python", "java", "sql", "apis", "databases", "docker", "git"]},
    {"title": "Frontend Developer",   "skills": ["javascript", "html", "css", "react", "ui", "git"]},
    {"title": "DevOps Engineer",      "skills": ["docker", "kubernetes", "aws", "ci/cd", "linux", "git", "automation"]},
    {"title": "Cloud Architect",      "skills": ["aws", "azure", "cloud", "docker", "kubernetes", "networking", "automation"]},
    {"title": "Data Engineer",        "skills": ["python", "sql", "spark", "hadoop", "etl", "databases", "aws"]},
    {"title": "Cybersecurity Analyst","skills": ["networking", "linux", "security", "python", "firewalls", "risk analysis"]},
    {"title": "AI Research Scientist","skills": ["python", "deep learning", "machine learning", "mathematics", "statistics", "research"]},
    {"title": "Full Stack Developer", "skills": ["javascript", "python", "react", "sql", "apis", "git", "databases"]},
    {"title": "Mobile Developer",     "skills": ["java", "kotlin", "swift", "ios", "android", "apis", "git"]},
    {"title": "Database Administrator","skills": ["sql", "databases", "oracle", "performance tuning", "backup", "linux"]},
]

# ─────────────────────────────────────────────
# STEP 1: TF-IDF VECTORIZER (manual)
# ─────────────────────────────────────────────

def compute_tfidf(job_roles):
    # Count how many documents each skill appears in
    doc_freq = defaultdict(int)
    total_docs = len(job_roles)

    for role in job_roles:
        unique_skills = set(role["skills"])
        for skill in unique_skills:
            doc_freq[skill] += 1

    # Build TF-IDF vectors for each role
    tfidf_vectors = []
    for role in job_roles:
        skills = role["skills"]
        total_terms = len(skills)
        vector = {}
        for skill in skills:
            tf = skills.count(skill) / total_terms
            idf = math.log(total_docs / doc_freq[skill])
            vector[skill] = tf * idf
        tfidf_vectors.append(vector)

    return tfidf_vectors, doc_freq, total_docs

# ─────────────────────────────────────────────
# STEP 2: COSINE SIMILARITY
# ─────────────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    # Dot product
    dot_product = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in vec_b)

    # Magnitudes
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (mag_a * mag_b)

# ─────────────────────────────────────────────
# STEP 3: BUILD USER PROFILE VECTOR
# ─────────────────────────────────────────────

def build_user_vector(user_skills, doc_freq, total_docs):
    total_terms = len(user_skills)
    vector = {}
    for skill in user_skills:
        skill = skill.lower().strip()
        tf = user_skills.count(skill) / total_terms
        # If skill not in dataset, give a small IDF
        idf = math.log(total_docs / doc_freq.get(skill, 1))
        vector[skill] = tf * idf
    return vector

# ─────────────────────────────────────────────
# STEP 4: RECOMMENDATION ENGINE (Score → Sort → Filter Top-N)
# ─────────────────────────────────────────────

def recommend(user_skills, top_n=3):
    tfidf_vectors, doc_freq, total_docs = compute_tfidf(job_roles)
    user_vector = build_user_vector(user_skills, doc_freq, total_docs)

    scores = []
    for i, role in enumerate(job_roles):
        score = cosine_similarity(user_vector, tfidf_vectors[i])
        scores.append((role["title"], round(score, 4)))

    # Sort descending
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_n]

# ─────────────────────────────────────────────
# MAIN: User Interface
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("   Tech Stack Recommender - DecodeLabs P3")
    print("=" * 50)
    print("Enter at least 3 skills (e.g. python, sql, aws)")
    print("Separate skills with commas.\n")

    raw_input_skills = input("Your skills: ")
    user_skills = [s.strip().lower() for s in raw_input_skills.split(",") if s.strip()]

    if len(user_skills) < 3:
        print("\n[ERROR] Please enter at least 3 skills.")
        return

    print(f"\nAnalyzing profile: {user_skills}")
    print("\nTop 3 Recommended Career Paths:")
    print("-" * 40)

    results = recommend(user_skills, top_n=3)
    for rank, (title, score) in enumerate(results, 1):
        bar = "█" * int(score * 40)
        print(f"{rank}. {title}")
        print(f"   Match Score: {score:.4f}  |{bar}|")
        print()

    print("=" * 50)
    print("Recommendation complete.")

if __name__ == "__main__":
    main()
