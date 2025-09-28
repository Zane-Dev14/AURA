
# AURA — MARL Autoscaler for Microservices

**Short summary:** AURA is a research platform to build a Multi-Agent RL autoscaler that coordinates horizontal scaling (replicas + optional soft resource limits) of a 3-tier microservice (frontend → backend → database). We train in simulation, validate locally on k3d, then run final experiments on a real Kubernetes cluster to compare MARL vs HPA/VPA baselines.

## Repo layout (phase-0)

```

AURA/
├── docker-compose.yml
├── infra/
│   ├── k3d-setup.md
│   ├── manifests/
│   │   └── three-tier/
│   └── helm/
├── microservices/
│   ├── three-tier/          
│   └── locust/
├── metrics/
│   ├── prometheus/
│   └── grafana/
├── simulator/
├── marl/
├── deployment/
├── experiments/
├── docs/
└── tools/
├── setup_project.sh
├── setup_k3d.sh
├── deploy_stack.sh
└── run_training.sh

````

---

## Phase-0: exact commands for teammates (run these **from the repo root**)

> These commands assume the repository is already cloned and you are in the repository root. Do **not** use absolute paths.

### Clone repo (one-time)
```bash
# from wherever you keep your code locally
git clone https://github.com/Zane-Dev14/AURA.git
cd AURA
````
### Create and start local k3d cluster & deploy (repository provides scripts)

```bash
# ensure tools are executable and run
chmod +x tools/setup_k3d.sh
./tools/setup_k3d.sh

# deploy manifests (3-tier app + monitoring)
chmod +x tools/deploy_stack.sh
./tools/deploy_stack.sh
```

**Check status**

```bash
kubectl get nodes
kubectl get pods --all-namespaces
kubectl get svc --all-namespaces
```

---

## Branching strategy & exact git workflow

We use a simple, safe workflow:

* `main` — production/stable; protected
* `dev` — integration branch; everything merges here first
* `feature/<short-name>` — personal feature branches

### How to start work (CLI)

```bash
# ensure you have latest dev
git fetch origin
git checkout dev
git pull origin dev

# create your feature branch from dev
git checkout -b feature/your-feature-name

# do work, then:
git add .
git commit -m "feat: short description of work"
git push -u origin feature/your-feature-name
```

### Open PR

* Open a Pull Request on GitHub: base = `dev`, compare = `feature/your-feature-name`
* Request reviewers and link related issue(s).

### Keep your branch up to date (recommended: rebase dev)

```bash
# fetch upstream changes
git fetch origin

# rebase your feature branch on latest dev
git checkout feature/your-feature-name
git rebase origin/dev

# if rebase succeeded:
git push --force-with-lease origin feature/your-feature-name
```

> **If you prefer merge instead of rebase**:

```bash
git checkout feature/your-feature-name
git merge origin/dev
# resolve conflicts if any, commit, then push
git push origin feature/your-feature-name
```

### Merging into `dev`

* After PR review, tests and approvals, merge PR (prefer **Squash and merge** to keep history clean).
* CI must pass before merging (we recommend enabling CI checks).

---

## Handling conflicts (step-by-step)

**If a rebase reveals conflicts:**

```bash
# you're mid-rebase and git stops with conflict
# open files, fix conflicts, then:
git add <fixed-files>
git rebase --continue

# if you want to abort:
git rebase --abort
```

**If you merged `dev` into your branch and see conflicts:**

* Resolve in your editor, `git add` resolved files, `git commit`, then `git push origin feature/...`.

**After rebasing, pushing requires force (safe):**

```bash
git push --force-with-lease origin feature/your-feature-name
```

`--force-with-lease` is safer than `--force` — it fails if upstream changed unexpectedly.

---

## GitHub Desktop — step by step (for non-CLI users)

1. Open GitHub Desktop and **File → Clone repository** → choose `AURA`.
2. From the **Current Branch** dropdown, select **New Branch**.

   * Base branch: `dev` (pull `dev` from origin first if not present).
   * Name: `feature/your-feature-name`.
3. Make edits in your code editor.
4. In GitHub Desktop: write commit message → **Commit to feature/your-feature-name**.
5. Click **Push origin** (top bar).
6. Click **Branch → Create pull request** (or open the PR on GitHub.com).
7. To update your branch with `dev`:

   * Fetch origin (Fetch origin button).
   * Switch to `dev` branch, click **Fetch origin** then **Pull**.
   * Switch to your feature branch → Branch → **Merge into current branch** → choose `dev` to merge in `dev` changes. Resolve conflicts via editor and commit.

---

## Collaborators, permissions & repository settings

### Add collaborators (owners / repo admins)

1. On GitHub.com: go to your repository → **Settings → Manage access** (or **Collaborators & teams**).
2. Click **Invite teams or people** → add teammates’ GitHub usernames or emails.
3. Choose role:

   * **Write** (can push branches and create PRs)
   * **Maintain** / **Admin** for more permissions (be careful)

### Fork model (alternative)

* Contributors fork the repo, push to their fork, and open PRs to `dev`.
* For forks, recommend adding the original repo as `upstream`:

```bash
git remote add upstream https://github.com/<ORG_OR_USER>/AURA.git
git fetch upstream
git checkout -b feature/xyz upstream/dev
```

### Protect critical branches

(Repository → Settings → Branches → Branch protection rules)

Recommended rules for `main` and `dev`:

* Require PR reviews before merge (1 or 2 reviewers)
* Require status checks / CI to pass
* Require up-to-date with base branch before merging
* Restrict who can push to `main` (disallow direct pushes)

Add a `CODEOWNERS` file in `.github/` if specific teams must review specific paths.

---

## Pull Request checklist (put in PR template)

* [ ] Code builds locally
* [ ] Lint and unit tests passed
* [ ] If changed service images, updated Dockerfile and tested image build
* [ ] If K8s manifests changed, validated `kubectl apply --dry-run=server -f` locally
* [ ] Add/Update documentation (README or infra docs)
* [ ] Assigned reviewer(s)

---

## Roles & responsibilities (concise)

* **DevOps / Infra (Parik)**

  * k3d scripts, cluster manifests, helm values, RBAC for agent
  * `tools/setup_k3d.sh`, `tools/deploy_stack.sh`

* **Microservices Dev (Vaish)**

  * Integrate 3-tier app, containerize, expose metrics endpoints
  * Locust load scripts

* **Metrics / Observability (brainrot)**

  * Prometheus scrape config, Grafana dashboards, PromQL evaluation queries

* **Simulator / Env (Parik)**

  * Simulator API that mimics K8s pod delays and scrape lag

* **ML / MARL (Muscle Man)**

  * PettingZoo env, QMIX trainer, checkpoints in `marl/policies/`

* **Agent Controller (Parik + ML)**

  * `deployment/agent-controller.py`, safe action clipping, cooldown, RBAC

* **Experiments / Evaluation (All)**

  * Run baselines (HPA/VPA), collect pod-hours, P95 latency; produce plots & CSVs

> Each feature task should be implemented on its own `feature/*` branch. Assign reviewers from the role most relevant to the change.

---

## How to keep local environment consistent

* Use `docker` + `k3d` locally (scripts provided in `tools/`)
* Store model checkpoints & experiment outputs in `experiments/results/` (persisted volume if running containers)
* Add `requirements.txt` for Python components, and pin versions in Dockerfiles

---

## Next steps (recommended immediate actions)

1. Pick one of the sample 3-tier apps linked above and add it under `microservices/three-tier` (submodule recommended).
2. Push `dev` branch to remote and protect `main` & `dev` in repo settings.
3. Add teammates as collaborators (or instruct them to fork).
4. Each teammate creates `feature/*` branches from `dev` and follow the workflow above.
5. Optionally: Add a simple GitHub Action that lints Python and builds Docker images on PRs.

---

## Contact / support

If anything fails during setup, include:

* `git status` output
* `kubectl get pods --all-namespaces`
* `./tools/setup_k3d.sh` logs (paste terminal output)
  Open an issue in the repo with that information and tag @Parik (Infra) or @Vaish (Microservices).

---
