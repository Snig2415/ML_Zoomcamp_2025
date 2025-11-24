import os
import tempfile
import subprocess
import argparse

# ===== CONFIGURATION =====
required_files = [
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "notebooks/notebook.ipynb"
]

optional_files = [
    "docker-compose.yml"
]

# ===== Helper Functions =====
def check_file(repo_path, file_path):
    full_path = os.path.join(repo_path, file_path)
    return os.path.exists(full_path)

def file_summary(repo_path):
    print(f"\nEvaluating repo at: {repo_path}\n")
    
    print("=== Required Files ===")
    for f in required_files:
        status = "✅ Found" if check_file(repo_path, f) else "❌ Missing"
        print(f"{status}  {f}")
    
    print("\n=== Optional Files ===")
    for f in optional_files:
        status = "✅ Found" if check_file(repo_path, f) else "⚠️ Missing"
        print(f"{status}  {f}")
    
    print("\n=== README Check ===")
    readme_path = os.path.join(repo_path, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content.split()) > 100:
            print("✅ README contains a detailed description")
        else:
            print("⚠️ README is too short or missing problem description")
    else:
        print("❌ README.md not found")

    print("\n=== Docker Check ===")
    if check_file(repo_path, "Dockerfile"):
        print("✅ Dockerfile exists")
    else:
        print("❌ Dockerfile missing")
    
    if check_file(repo_path, "docker-compose.yml"):
        print("✅ docker-compose.yml exists")
    else:
        print("⚠️ docker-compose.yml missing (optional for multi-service setup)")

# ===== Main =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local GitHub repo scorer")
    parser.add_argument("https://github.com/Fharuk/DATATALKS-2025-COHORT.git", help="GitHub repository URL to evaluate")
    args = parser.parse_args()

    # repo_url = args.repo_url
    repo_url = args.repo

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Cloning {repo_url} into temporary folder...")
        try:
            subprocess.run(["git", "clone", repo_url, tmpdir], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to clone the repository. Make sure 'git' is installed and URL is correct.")
            exit(1)
        
        # Run file checks on cloned repo
        file_summary(tmpdir)
