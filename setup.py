# Confirm you're in the correct directory before we begin installation
import os
import subprocess
import sys
import getpass

project_root = os.getcwd()
print(project_root) # output below should be /Users/dahnayajoyner/RAG/GenAI-Dev-Onboarding-Starter-Kit/

# Check Poetry version
try:
    result = subprocess.run(["poetry", "--version"], capture_output=True, text=True, check=True)
    print(f"Poetry version: {result.stdout.strip()}")
    print("Poetry is installed and working.")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Poetry is not installed or not in PATH.")
    print("Please install Poetry manually using: curl -sSL https://install.python-poetry.org | python3 -")

# Get API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    # Request for new OpenAI API key if none available
    api_key = getpass.getpass("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

if os.environ.get("OPENAI_API_KEY"):
    print("OpenAI API key set successfully!")
else:
    print("Failed to set OpenAI API key.")