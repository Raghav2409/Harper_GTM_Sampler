# Fixing GitHub Push Error 403

## The Problem
You're getting a 403 error because either:
1. The repository doesn't exist on GitHub yet, OR
2. You're using your GitHub password instead of a Personal Access Token

## Solution

### Step 1: Create the Repository on GitHub
**This is required before you can push!**

1. Go to: https://github.com/new
2. Repository name: `Harper_GTM_Sampler`
3. Description: "GTM Co-Pilot Dashboard for AI-native commercial insurance brokerage"
4. Choose Public or Private
5. **IMPORTANT:** Do NOT check "Add a README file", "Add .gitignore", or "Choose a license"
   - We already have these files locally
6. Click "Create repository"

### Step 2: Create a Personal Access Token
GitHub no longer accepts passwords for Git operations. You MUST use a token:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Note: "Harper GTM Sampler"
4. Expiration: Choose your preference (90 days, 1 year, etc.)
5. Select scopes: Check **`repo`** (this gives full repository access)
6. Click "Generate token" at the bottom
7. **COPY THE TOKEN IMMEDIATELY** - you won't see it again!
   - It will look like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Step 3: Push Using the Token
```bash
git push -u origin main
```

When prompted:
- **Username:** `Raghav2409`
- **Password:** Paste your Personal Access Token (the `ghp_...` token, NOT your GitHub password)

### Alternative: Use Token in URL (One-time)
If you want to avoid entering credentials each time:
```bash
git remote set-url origin https://Raghav2409:YOUR_TOKEN_HERE@github.com/Raghav2409/Harper_GTM_Sampler.git
git push -u origin main
```
(Replace YOUR_TOKEN_HERE with your actual token)

## Quick Checklist
- [ ] Repository created on GitHub
- [ ] Personal Access Token created with `repo` scope
- [ ] Token copied and ready to paste
- [ ] Run `git push -u origin main`
- [ ] Use token as password (not GitHub password)
