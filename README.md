# 1. Set your Git identity
git config --global user.name "Gourav Singh"
git config --global user.email "gouravsingh1604@gmail.com"

# 2. Initialize the repo (if not already)
git init

# 3. Stage all files
git add .

# 4. Make the initial commit
git commit -m "Initial commit"

# 5. Rename the branch to main
git branch -M main

# 6. Add the remote (replace URL if needed)
git remote add origin https://github.com/gouravsingh1604/OS-ALGORITHM-SIMULATOR.git

# 7. Push to GitHub
git push -u origin main
