# FAHM Mammogram Classifier — Step-by-Step Deployment Guide 🚀

This document provides clear, step-by-step instructions on how to make your code changes live when modifying the FAHM Mammogram Classifier application.

---

## 🏗️ Architecture Overview

To leverage maximum performance and zero-cost hosting, the application is divided into two decoupled services:
1. **Frontend UI (Vercel):** Hosts the interactive clinician portal built with React + Vite.
2. **Backend API (Hugging Face Spaces):** Runs the FastAPI application inside a Docker container containing the PyTorch ML model (`EfficientNet-B0`).

Because they are hosted on different platforms, changes to the frontend and backend are deployed separately.

---

## 🎨 1. How to Deploy Frontend Changes (Vercel)

*Use these steps if you made changes to the files in the `frontend/` directory (e.g., UI styles, buttons, pages, logic, text).*

### Step 1: Open Your Terminal
Open your terminal (PowerShell, Command Prompt, or Git Bash) and navigate to the project root directory:
```bash
cd d:\fahm
```

### Step 2: Stage and Commit the Changes
Stage the frontend folder changes and commit them with a descriptive message:
```bash
git add frontend/
git commit -m "Update frontend: [describe what you changed]"
```

### Step 3: Push to Technical Assessment Repository
Push the commit to your main technical assessment GitHub repository:
```bash
git push origin main
```

### Step 4: Deploy to Vercel (Live App)
The live Vercel website is connected to your frontend-only repository (`fahm-mammogram-classifier`). To push the `frontend/` subdirectory changes to that repository and trigger the Vercel build, run the following commands in order:
```bash
# 1. Create a local temporary branch containing only the frontend/ folder contents
git subtree split --prefix=frontend -b temp-deploy-branch

# 2. Force-push this temporary branch to the live Vercel repository
git push https://github.com/adeelshah41/fahm-mammogram-classifier.git temp-deploy-branch:main --force

# 3. Delete the temporary branch
git branch -D temp-deploy-branch
```

### Step 5: Verification
* **Vercel Auto-Deployment:** Vercel will detect the push to `fahm-mammogram-classifier`, trigger the build, and deploy the update in 1–2 minutes.
* **Check Live Site:** Open [https://fahm-mammogram-classifier.vercel.app](https://fahm-mammogram-classifier.vercel.app) to verify the changes.

---

## 🧠 2. How to Deploy Backend & Model Changes (Hugging Face)

*Use these steps if you modified files in `backend/` (API routes, services, schemas), retrained a model in `model/` (generating a new `model.pt`), or made changes to the Docker configuration.*

### Step 1: Sync Changes with the `hf-deploy` Folder
Hugging Face Spaces requires the `Dockerfile` and the backend app to be at the **root** of the deployment directory. To do this, we use the `hf-deploy` folder as our staging area:

1. **If you changed backend code:**
   Copy the updated files from `d:\fahm\backend\app\` to `d:\fahm\hf-deploy\app\`.
2. **If you trained a new model:**
   Copy the new `model.pt` weights file to `d:\fahm\hf-deploy\app\models\model.pt`.

---

### Step 2: Push to Hugging Face Spaces

You have **two options** to deploy the changes to Hugging Face Spaces:

#### 🔹 Option A: Direct Git Push using Git Subtree (Recommended)
This is the fastest method. It pushes only the contents of the `hf-deploy` folder directly to Hugging Face.

1. Open your terminal at the project root (`d:\fahm`).
2. Run this single command:
   ```bash
   git subtree push --prefix hf-deploy https://huggingface.co/spaces/adeel41/Mammogram-Classifier main
   ```
3. Enter your Hugging Face credentials (Username and Access Token/Password) if prompted.

#### 🔹 Option B: Manual Web Upload (Easiest, No Command Line)
If you prefer a visual interface without running command line git pushes:

1. Open your web browser and go to your [Hugging Face Space Files](https://huggingface.co/spaces/adeel41/Mammogram-Classifier/tree/main).
2. Click on **Files and versions** at the top.
3. Locate the file you updated (e.g. `app/main.py` or `app/models/model.pt`).
4. Click **Add file** -> **Upload files**.
5. Drag and drop the modified files from your local `d:\fahm\hf-deploy\` folder into the web browser interface.
6. Scroll down, add a commit message (e.g. *"Update model weights"*), and click **Commit changes to main**.

---

### Step 3: Verification
* **Hugging Face Auto-Rebuild:** Hugging Face will immediately start rebuilding the Docker container. This process takes about 2–3 minutes.
* **Monitor Build Status:** You can watch the real-time build logs at: [https://huggingface.co/spaces/adeel41/Mammogram-Classifier](https://huggingface.co/spaces/adeel41/Mammogram-Classifier).
* **Test the API:** Verify the live API docs at [https://adeel41-mammogram-classifier.hf.space/docs](https://adeel41-mammogram-classifier.hf.space/docs).

---

## 🔄 3. Keep the Main GitHub Repo in Sync (Important)

After you deploy your backend changes to Hugging Face, make sure to commit and push the updated files in `backend/` and `hf-deploy/` to your main GitHub repository so that your codebase remains unified and complete:

```bash
cd d:\fahm
git add backend/ hf-deploy/ docs/
git commit -m "Sync backend and deployment files with Hugging Face Space"
git push origin master
```

---

## 💡 Troubleshooting & Pro Tips

* **Slow First Request?** If the backend API hasn't been accessed in 15+ minutes, Hugging Face automatically puts the container to sleep to save resources. The first request might take 30–60 seconds to wake it up. Subsequent requests will execute in 1–2 seconds.
* **Model Size Limit:** The `model.pt` weights are around 16.3MB. Since this is well under Hugging Face's 100MB soft-limit for individual files, it can be pushed directly without needing Git LFS (Large File Storage).
* **Frontend-Backend Sync:** If you ever change the backend API endpoints (e.g. adding new query parameters or changing path structures), remember to update the corresponding API calling URLs in `frontend/src/` to match.
