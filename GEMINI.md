# Workspace Policy & Guidelines

## Git Commit Policy (Mandatory)
Every completed task, feature, data pipeline update, or code change in this repository must be committed to Git immediately upon completion.

### Rules:
1. **Always Commit Completed Tasks:**
   - As soon as a task, feature, fix, or data generation step is finished and verified, create a Git commit before moving on or concluding the turn.
2. **Conventional Commits Standard:**
   - Commit messages must follow the `type(optional-scope): message` format:
     - `feat(...)`: New features, pipelines, models, apps, or data products.
     - `fix(...)`: Bug fixes or corrections in data/models/code.
     - `data(...)` or `feat(data-eng)`: Datasets, schemas, ETL pipelines, or metadata files.
     - `docs(...)`: Documentation, README, guides, and slide summaries.
     - `chore(...)`: Config, `.gitignore`, dependencies, or maintenance.
     - `refactor(...)`: Code improvements without functionality changes.
3. **Granularity & Quality:**
   - Commits should be granular and logically isolated per step.
   - Never commit sensitive secrets, `.env` files, or raw large model weights (`*.pt`).
