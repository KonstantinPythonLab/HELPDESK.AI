"""
Comprehensive Contributor Compliance Audit for all open Issues & PRs
"""

import sys
import json
import subprocess

REPO = "riteshbonthalakoti/HELPDESK.AI"
ADMIN = "riteshbonthalakoti"

def run_gh_json(cmd):
    try:
        res = subprocess.run(["gh"] + cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        return json.loads(res.stdout) if res.stdout else []
    except Exception as e:
        print(f"[ERROR] gh {' '.join(cmd)}: {e}", file=sys.stderr)
        return []

def get_paginated(endpoint):
    items = []
    page = 1
    while True:
        res = run_gh_json(["api", f"{endpoint}?per_page=100&page={page}"])
        if not res or not isinstance(res, list):
            break
        items.extend(res)
        if len(res) < 100:
            break
        page += 1
    return items

def main():
    print("Loading followers...")
    followers_data = get_paginated(f"/users/{ADMIN}/followers")
    followers = {u["login"].lower() for u in followers_data if "login" in u}

    print("Loading stargazers...")
    stargazers_data = get_paginated(f"/repos/{REPO}/stargazers")
    stargazers = {u["login"].lower() for u in stargazers_data if "login" in u}

    print("Loading forks...")
    forks_data = get_paginated(f"/repos/{REPO}/forks")
    forkers = {f["owner"]["login"].lower() for f in forks_data if "owner" in f and "login" in f["owner"]}

    print(f"Stats: {len(followers)} followers, {len(stargazers)} stargazers, {len(forkers)} forks.\n")

    # Fetch open issues
    issues = run_gh_json(["issue", "list", "--repo", REPO, "--state", "open", "--limit", "100",
                          "--json", "number,title,author,comments"])
    print(f"=== Open Issues ({len(issues)}) ===")
    issues_to_remind = []
    for issue in issues:
        author = issue["author"]["login"]
        if author.lower() in (ADMIN.lower(), "github-actions", "dependabot", "app/dependabot"):
            continue

        has_owner_comment = any(
            c.get("author", {}).get("login", "").lower() == ADMIN.lower()
            for c in issue.get("comments", [])
        )

        u = author.lower()
        is_foll = u in followers
        is_star = u in stargazers
        is_fork = u in forkers

        if not (is_foll and is_star and is_fork):
            issues_to_remind.append({
                "number": issue["number"],
                "author": author,
                "title": issue["title"],
                "has_owner_comment": has_owner_comment,
                "is_following": is_foll,
                "has_starred": is_star,
                "has_forked": is_fork
            })
            commented_str = "[ALREADY COMMENTED]" if has_owner_comment else "[NEEDS COMMENT]"
            print(f"Issue #{issue['number']:<4} by @{author:<18} {commented_str} | Follow: {is_foll}, Star: {is_star}, Fork: {is_fork}")

    print(f"\nTotal issues needing compliance action: {len([i for i in issues_to_remind if not i['has_owner_comment']])}")

if __name__ == "__main__":
    main()
