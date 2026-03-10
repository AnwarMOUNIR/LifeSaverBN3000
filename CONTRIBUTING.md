# Contributing to LifeSaverBN3000

Thank you for being part of the **LifeSaverBN3000** Medical Decision Support Application team! 

As per our `team_organization.md` rules, please follow these guidelines to assure smooth collaboration.

## 1. Forking vs. Direct Branching
We do **not** use forks for this internal team project. All work is done on branches within the main repository.

## 2. Cloning the Repository
First, clone the main repository to your local machine:
```bash
git clone <repository_url>
cd LifeSaverBN3000
```

## 3. Creating a Branch
**Never commit directly to the `main` branch.** Branch protection rules are enforced.
When starting a new Jira task, create a branch with a descriptive name indicating the feature/fix (e.g., `feat/`, `fix/`, `docs/`):

```bash
git checkout -b feat/your_feature_name
```

## 4. Local Development
1. Create and activate a Virtual Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. Install the shared dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make your code changes following PEP8 standards.

## 5. Testing Your Code
Before committing, ensure your code passes local tests. 
- For standard python components, run: `pytest`
- To verify the UI works: `streamlit run app/app.py`
- If you introduced new dependencies, make sure you ran `pip freeze > requirements.txt` (or manually added them).

## 6. Committing and Pushing
Write meaningful, concise commit messages.
```bash
git add .
git commit -m "feat: added SHAP dependency to requirements and tests"
git push origin feat/your_feature_name
```

## 7. Opening a Pull Request (PR)
1. Go to the GitHub repository online.
2. Click **"Compare & pull request"**.
3. Fill out the PR description template clearly stating what was resolved (link any Jira tickets).
4. **Tag at least one team member for a Code Review.**
5. Wait for the automated GitHub Actions (CI/CD pytest) to pass successfully.
6. Once reviewed and tests pass, merge the PR into `main` and delete your local/remote feature branch.
