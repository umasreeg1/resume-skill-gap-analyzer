from typing import List, Dict

# Pre-defined domain recommendations dictionary
RECOMMENDATION_RULES = {
    # AI/ML & Data Science
    "TensorFlow": "Gain practical experience building and training neural network models with TensorFlow.",
    "PyTorch": "Practice deep learning model development, custom training loops, and PyTorch tensors.",
    "Deep Learning": "Study deep learning architecture fundamentals including CNNs, RNNs, and Transformers.",
    "Machine Learning": "Strengthen core machine learning concepts (supervised, unsupervised algorithms, model evaluation).",
    "NLP": "Explore NLP techniques including text preprocessing, tokenization, TF-IDF, and word embeddings.",
    "Natural Language Processing": "Explore NLP techniques including text preprocessing, tokenization, TF-IDF, and word embeddings.",
    "Computer Vision": "Practice image processing fundamentals and computer vision models using OpenCV or PyTorch.",
    "Scikit-learn": "Work on end-to-end ML classification and regression projects using Scikit-learn pipelines.",
    "OpenCV": "Build hands-on computer vision mini-projects for image recognition and object detection.",

    # Data & Databases
    "SQL": "Improve SQL querying skills, database schema design, indexing, and multi-table JOIN operations.",
    "MySQL": "Practice MySQL database administration, relational query optimization, and transaction management.",
    "PostgreSQL": "Explore PostgreSQL advanced features, JSON storage, and spatial query capabilities.",
    "MongoDB": "Learn NoSQL database concepts, document modeling, and aggregation pipelines with MongoDB.",
    "Power BI": "Build interactive business dashboards and data visualizations using Microsoft Power BI.",
    "Tableau": "Learn data visualization, storyboarding, and reporting using Tableau Desktop.",
    "Pandas": "Practice data cleaning, aggregation, reshaping, and exploratory analysis with Pandas DataFrames.",
    "Apache Spark": "Learn distributed big data processing and PySpark for large-scale data analytics.",

    # Cloud & DevOps
    "AWS": "Consider learning AWS cloud fundamentals, IAM, S3 storage, EC2 compute, and serverless Lambda.",
    "Azure": "Explore Microsoft Azure core cloud services, Virtual Machines, and Azure App Services.",
    "Google Cloud": "Gain familiarity with Google Cloud Platform (GCP) services like Compute Engine and BigQuery.",
    "GCP": "Gain familiarity with Google Cloud Platform (GCP) services like Compute Engine and BigQuery.",
    "Docker": "Practice containerizing applications with Dockerfiles and orchestrating services via Docker Compose.",
    "Kubernetes": "Explore container orchestration, pod management, and deployment configurations with Kubernetes.",
    "CI/CD": "Set up automated CI/CD pipelines using GitHub Actions, GitLab CI, or Jenkins.",

    # Web Development
    "React": "Build interactive user interfaces using React components, state management (Hooks), and JSX.",
    "React.js": "Build interactive user interfaces using React components, state management (Hooks), and JSX.",
    "Node.js": "Develop server-side backend RESTful web services using Node.js and Express.",
    "Flask": "Practice building lightweight Python web APIs and microservices using Flask.",
    "FastAPI": "Build high-performance asynchronous REST APIs with auto-generated docs using FastAPI.",
    "Django": "Explore full-stack web development using Django's ORM, authentication, and admin panel.",

    # Programming
    "Python": "Strengthen Python object-oriented programming, data structures, and standard library modules.",
    "Java": "Review Core Java fundamentals, OOP concepts, multithreading, and Spring Boot basics.",
    "C++": "Practice C++ memory management, pointers, and Standard Template Library (STL) algorithms.",
    "Go": "Explore Golang concurrency patterns (goroutines, channels) and microservice development.",
    "Rust": "Study Rust ownership model, memory safety, and high-performance system programming.",

    # Tools & Concepts
    "Git": "Improve Git workflow skills including branching, merging, pull requests, and merge conflict resolution.",
    "GitHub": "Practice hosting projects, collaborating via Pull Requests, and managing repositories on GitHub.",
    "System Design": "Study scalable system architecture, load balancing, caching strategies, and database sharding.",
    "Agile": "Familiarize yourself with Agile/Scrum software development lifecycles and sprint management."
}


def generate_recommendations(missing_skills: List[str]) -> Dict[str, str]:
    """
    Generates structured, rule-based learning recommendations for missing skills.
    
    Returns:
        dict: Mapping of skill -> recommendation string.
    """
    if not missing_skills:
        return {
            "Great Job!": "Your resume already covers all the key technical skills specified in the job description."
        }

    recommendations = {}
    for skill in missing_skills:
        if skill in RECOMMENDATION_RULES:
            recommendations[skill] = RECOMMENDATION_RULES[skill]
        else:
            recommendations[skill] = f"Consider building a mini-project or taking a quick tutorial to demonstrate practical skill in {skill}."

    return recommendations
