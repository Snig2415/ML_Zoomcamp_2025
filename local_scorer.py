import os

# ===== CONFIGURATION =====
repo_path = r"https://github.com/K-Opoku/fraud-detection-service" #C:\Users\snigd\MLZoomcamp2025_Midterm1"  # Change to your repo path
required_files = [
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "notebooks/notebook.ipynb"
]

optional_files = [
    "docker-compose.yml"
]

# ===== HELPER FUNCTIONS =====
def check_file(file_path):
    full_path = os.path.join(repo_path, file_path)
    return os.path.exists(full_path)

def file_summary():
    print(f"\nEvaluating repo at: {repo_path}\n")
    
    print("=== Required Files ===")
    for f in required_files:
        status = "✅ Found" if check_file(f) else "❌ Missing"
        print(f"{status}  {f}")
    
    print("\n=== Optional Files ===")
    for f in optional_files:
        status = "✅ Found" if check_file(f) else "⚠️ Missing"
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
    if check_file("Dockerfile"):
        print("✅ Dockerfile exists")
    else:
        print("❌ Dockerfile missing")
    
    if check_file("docker-compose.yml"):
        print("✅ docker-compose.yml exists")
    else:
        print("⚠️ docker-compose.yml missing (optional for multi-service setup)")

# ===== RUN SCRIPT =====
if __name__ == "__main__":
    file_summary()
