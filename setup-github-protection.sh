#!/usr/bin/env bash
set -euo pipefail

# Configures branch-protection ruleset and repo-level settings via GitHub API.
# This is in addition to the org-level ruleset which already covers deletion
# prevention, approvers, stale reviews, code-owner review, and last-push approval.
#
# Requires: gh CLI with admin access to the repo.

RULESET_NAME="protect_main"
REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
REPO="$(printf '%s' "$REMOTE_URL" | sed -E 's#.*[:/]([^/]+/[^/]+)$#\1#' | sed 's/\.git$//')"
[[ -z "$REPO" ]] && { echo "Error: cannot detect repo from git remote." >&2; exit 1; }

DRY_RUN=false
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        echo "Usage: $(basename "$0") [--dry-run]" >&2
        exit 1
    fi
fi

echo "Repository: $REPO"
echo ""

# Preflight
command -v gh &>/dev/null || { echo "Error: gh CLI not installed." >&2; exit 1; }
command -v jq &>/dev/null || { echo "Error: jq not installed." >&2; exit 1; }
gh auth status &>/dev/null || { echo "Error: gh not authenticated. Run 'gh auth login'." >&2; exit 1; }

if [[ "$DRY_RUN" == false ]]; then
    [[ "$(gh api "repos/$REPO" --jq '.permissions.admin' 2>/dev/null)" == "true" ]] \
        || { echo "Error: no admin access to '$REPO'." >&2; exit 1; }
fi

# --- Payloads ---

CHECKS='[{"context":"Trunk Check"},{"context":"validate-jira-ticket / hygiene-check"}]'

# RepositoryRole actor_id 5 = built-in "Admin" role (1=Read..5=Admin)
RULESET_PAYLOAD="$(jq -n --arg name "$RULESET_NAME" --argjson checks "$CHECKS" '{
  name: $name,
  target: "branch",
  enforcement: "active",
  conditions: { ref_name: { include: ["~DEFAULT_BRANCH"], exclude: [] } },
  rules: [
    { type: "pull_request", parameters: {
        required_approving_review_count: 1,
        dismiss_stale_reviews_on_push: true,
        require_code_owner_review: true,
        require_last_push_approval: true,
        required_review_thread_resolution: true,
        required_reviewers: [],
        allowed_merge_methods: ["squash"]
    }},
    { type: "required_signatures" },
    { type: "required_status_checks", parameters: {
        required_status_checks: $checks,
        strict_required_status_checks_policy: true
    }}
  ],
  bypass_actors: [
    { actor_type: "OrganizationAdmin", bypass_mode: "always" },
    { actor_id: 5, actor_type: "RepositoryRole", bypass_mode: "always" }
  ]
}')"

SETTINGS_PAYLOAD='{
  "delete_branch_on_merge": true,
  "allow_squash_merge": true,
  "allow_merge_commit": false,
  "allow_rebase_merge": false
}'

# --- Dry run ---

if [[ "$DRY_RUN" == true ]]; then
    echo "=== Ruleset: POST repos/$REPO/rulesets ==="
    echo "$RULESET_PAYLOAD" | jq .
    echo ""
    echo "=== Settings: PATCH repos/$REPO ==="
    echo "$SETTINGS_PAYLOAD" | jq .
    echo ""
    echo "Dry run complete. Run without --dry-run to apply."
    exit 0
fi

# --- Apply ---

EXISTING_ID="$(gh api "repos/$REPO/rulesets" --jq \
    ".[] | select(.name == \"$RULESET_NAME\" and .source_type == \"Repository\") | .id" 2>/dev/null || true)"

if [[ -n "$EXISTING_ID" ]]; then
    echo "Updating ruleset '$RULESET_NAME' (id: $EXISTING_ID)..."
    gh api "repos/$REPO/rulesets/$EXISTING_ID" --method PUT --input - <<< "$RULESET_PAYLOAD" >/dev/null
else
    echo "Creating ruleset '$RULESET_NAME'..."
    gh api "repos/$REPO/rulesets" --method POST --input - <<< "$RULESET_PAYLOAD" >/dev/null
fi

echo "Applying repo settings..."
gh api "repos/$REPO" --method PATCH --input - <<< "$SETTINGS_PAYLOAD" >/dev/null

echo "Done. Verify at: https://github.com/$REPO/settings/rules"
