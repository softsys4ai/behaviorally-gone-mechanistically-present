#!/usr/bin/env bash
# Incremental, secret-safe backup to GitHub and Google Drive (rclone remote "gdrive:").
#
#   bash code/scripts/backup_live.sh            # incremental: git commit+push of code, rclone copy of results/notes
#   bash code/scripts/backup_live.sh --final    # + git bundle, env snapshot, rclone check, FINAL_BACKUP_OK marker
#
# Drive layout (rclone copy only — never deletes anything remote):
#   gdrive:${DRIVE_ROOT}/experiments/   /home/cc/experiments (files containing a secret are skipped and listed)
#   gdrive:${DRIVE_ROOT}/notes/         RESEARCH_CONTEXT.md, reports, recipes, plans, memory (secrets redacted)
#   gdrive:${DRIVE_ROOT}/claude/        Claude Code transcripts of this project (secrets redacted)
#   gdrive:${DRIVE_ROOT}/repo/          git bundle of all branches (--final)
#   gdrive:${DRIVE_ROOT}/env/           pip freeze, versions (--final)
# Secrets: the exact values of *TOKEN* / *API_KEY* / *SECRET* / *PASSWORD* variables in launch_deepseek_owl.sh
# and /home/cc/my_secrets/*.env are read into a
# mode-600 file (never printed) and redacted / used to skip files. Files with a PEM private key are never uploaded.
# GitHub: only code/{src,scripts,test,input} are committed (author per project convention), then pushed.
set -uo pipefail

FINAL=0; [ "${1:-}" = "--final" ] && FINAL=1
REPO=/home/cc/Subliminal-Steering-2026-Code
DRIVE_ROOT="${DRIVE_ROOT:-subliminal_live}"
ORCH=/home/cc/experiments/subliminal/orchestrator
STAGE="${ORCH}/backup_stage"
LOCK="${ORCH}/backup.lock"
mkdir -p "${ORCH}"
exec 9>"${LOCK}"
flock 9                                   # one backup at a time
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[backup $(ts)] start (final=${FINAL})"

# ── secrets list (values only, never echoed) ──
TOK="$(mktemp -p "${ORCH}" .tokens.XXXXXX)"; chmod 600 "${TOK}"
trap 'rm -f "${TOK}"' EXIT
/home/cc/llm_env/bin/python - "${TOK}" <<'EOF'
import re, sys, glob, os
vals = set()
for path in ["/home/cc/Subliminal-Steering-2026-Code/launch_deepseek_owl.sh"] + glob.glob("/home/cc/my_secrets/*.env"):
    if not os.path.exists(path):
        continue
    for line in open(path, errors="ignore"):
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip())
        if m and re.search(r"TOKEN|API_KEY|SECRET|PASSWORD|PRIVATE", m.group(1)):
            v = m.group(2).strip().strip('"').strip("'")
            if len(v) >= 16 and "$" not in v:
                vals.add(v)
open(sys.argv[1], "w").write("\n".join(sorted(vals)) + ("\n" if vals else ""))
EOF

redact_copy() {   # redact_copy <src> <dst>  — copy text with every secret value replaced; skip PEM keys
  /home/cc/llm_env/bin/python - "$1" "$2" "${TOK}" <<'EOF'
import os, sys, shutil
src, dst, tok = sys.argv[1:4]
secrets = [s for s in open(tok).read().split("\n") if s]
os.makedirs(os.path.dirname(dst), exist_ok=True)
data = open(src, "rb").read()
if b"PRIVATE KEY-----" in data:
    print(f"  skip (private key): {src}"); sys.exit(0)
for s in secrets:
    data = data.replace(s.encode(), b"<REDACTED>")
open(dst, "wb").write(data)
EOF
}

# ── GitHub: code only ──
cd "${REPO}"
if [ -n "$(git status --porcelain -- code/src code/scripts code/test code/input)" ]; then
  if git diff --cached --quiet && ! grep -rqF -f "${TOK}" code/src code/scripts code/test code/input 2>/dev/null; then
    git add code/src code/scripts code/test code/input
    git -c user.name="Mohammad Gholizadeh" -c user.email="msghik@yahoo.com" commit -q \
      -m "Auto-backup of code ($(ts))" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
      && echo "  git: committed $(git log --oneline -1)"
  else
    echo "  git: NOT committing (staged changes present or a secret found in code/) — check manually"
  fi
fi
git push -q origin "$(git rev-parse --abbrev-ref HEAD)" 2>&1 | tail -2 && echo "  git: pushed $(git rev-parse --short HEAD)"

# ── Drive: experiments (skip files that contain a secret) ──
EXCL="${ORCH}/excluded_secret_files.txt"
( cd /home/cc/experiments && grep -rlF -f "${TOK}" . 2>/dev/null | sed 's|^\./||' ) > "${EXCL}" || true
[ -s "${EXCL}" ] && echo "  WARNING: $(wc -l < "${EXCL}") experiment file(s) contain a secret and are not uploaded: ${EXCL}"
rclone copy /home/cc/experiments "gdrive:${DRIVE_ROOT}/experiments" --exclude-from "${EXCL}" \
  --exclude "subliminal/orchestrator/backup_stage/**" --exclude "subliminal/orchestrator/.tokens.*" \
  --exclude "subliminal/orchestrator/backup.lock" --transfers 8 --checkers 16 2>&1 | tail -3
echo "  drive: experiments copied"

# ── Drive: notes + Claude state (redacted copies) ──
rm -rf "${STAGE}"; mkdir -p "${STAGE}/notes" "${STAGE}/claude"
for f in /home/cc/RESEARCH_CONTEXT.md /home/cc/RESTORE_REPORT_2026-10-24.md /home/cc/GH200_SETUP_RECIPE.md \
         /home/cc/README.md /home/cc/2604.md; do
  [ -f "$f" ] && redact_copy "$f" "${STAGE}/notes/$(basename "$f")"
done
for d in /home/cc/docs /home/cc/.claude/plans /home/cc/.claude/projects/-home-cc/memory; do
  [ -d "$d" ] && find "$d" -type f | while read -r f; do redact_copy "$f" "${STAGE}/notes/$(basename "$d")/${f#$d/}"; done
done
find /home/cc/.claude/projects/-home-cc -maxdepth 1 -type f -name '*.jsonl' | while read -r f; do
  redact_copy "$f" "${STAGE}/claude/projects/-home-cc/$(basename "$f")"
done
if grep -rqF -f "${TOK}" "${STAGE}" 2>/dev/null; then
  echo "  ERROR: a secret survived redaction in the stage — notes/claude NOT uploaded"
else
  rclone copy "${STAGE}/notes" "gdrive:${DRIVE_ROOT}/notes" --transfers 8 2>&1 | tail -2
  rclone copy "${STAGE}/claude" "gdrive:${DRIVE_ROOT}/claude" --transfers 8 2>&1 | tail -2
  echo "  drive: notes + claude copied"
fi

# ── final extras ──
if [ "${FINAL}" = "1" ]; then
  mkdir -p "${STAGE}/repo" "${STAGE}/env"
  git -C "${REPO}" bundle create "${STAGE}/repo/Subliminal-Steering-2026-Code.bundle" --all 2>/dev/null
  git -C "${REPO}" log --oneline -20 > "${STAGE}/repo/git_log.txt"
  /home/cc/llm_env/bin/pip freeze > "${STAGE}/env/pip_freeze_llm_env.txt" 2>/dev/null
  { uname -a; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; /home/cc/llm_env/bin/python -c \
    "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"; } \
    > "${STAGE}/env/versions.txt" 2>&1
  rclone copy "${STAGE}/repo" "gdrive:${DRIVE_ROOT}/repo" 2>&1 | tail -2
  rclone copy "${STAGE}/env" "gdrive:${DRIVE_ROOT}/env" 2>&1 | tail -2
  if rclone check /home/cc/experiments "gdrive:${DRIVE_ROOT}/experiments" --one-way --exclude-from "${EXCL}" \
       --exclude "subliminal/orchestrator/**" 2>"${ORCH}/rclone_check.txt"; then
    echo "$(ts) final backup verified (rclone check: 0 differences)" | tee "${ORCH}/FINAL_BACKUP_OK"
    rclone copy "${ORCH}/FINAL_BACKUP_OK" "gdrive:${DRIVE_ROOT}/" 2>&1 | tail -1
  else
    echo "  ERROR: rclone check found differences — see ${ORCH}/rclone_check.txt"
  fi
fi
rclone copy "${ORCH}" "gdrive:${DRIVE_ROOT}/orchestrator" --exclude "backup_stage/**" --exclude ".tokens.*" \
  --exclude "backup.lock" 2>&1 | tail -1
echo "[backup $(ts)] done"
