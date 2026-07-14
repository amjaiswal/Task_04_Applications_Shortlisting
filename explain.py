def generate_explanation(student_name, student_skills, job_title, required_skills, score):
    """
    Generate human-readable explanation for a student-job match.
    """

    student_skill_set = {
        skill.strip().lower()
        for skill in student_skills.split(",")
    }

    job_skill_set = {
        skill.strip().lower()
        for skill in required_skills.split(",")
    }

    matched = student_skill_set.intersection(job_skill_set)
    missing = job_skill_set - student_skill_set

    explanation = {
        "student": student_name,
        "job": job_title,
        "match_score": round(score, 2),
        "matched_skills": list(matched),
        "missing_skills": list(missing),
        "reason": (
            f"{student_name} matches the role '{job_title}' "
            f"because {len(matched)} required skills are present."
        )
    }

    return explanation