import requests
import json

github_token = "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"

def call():
    # Create a request to the GitHub API to get a list of all the repositories under organization "nota-github"
    response=requests.get("https://api.github.com/orgs/nota-github/repos", headers={"Authorization": f"token {github_token}"})
    r = json.loads(response.text)
    print(len(r))

if __name__ == "__main__":
    call()
