#!/usr/bin/env bash
# Push the local bookmark database to the VPS.
#
# The local machine is the only writer. It holds the X cookies and it runs the
# refresh. The VPS is a read copy for browsing. Each push replaces the remote
# file, so a category edit or a settings change made on the VPS is lost.
#
# Usage: ./scripts/push-to-vps.sh
set -euo pipefail

SSH_HOST="vps-rsync"          # not "vps" — that host has a RemoteCommand
REMOTE_DIR="/root/projects/siftly"
SERVICE="siftly"
LOCAL_DB="prisma/dev.db"

if [ ! -f "$LOCAL_DB" ]; then
  echo "error: $LOCAL_DB not found. Run from the project root." >&2
  exit 1
fi

TMP_DB="$(mktemp -t siftly-push-XXXXXX.db)"
trap 'rm -f "$TMP_DB" "$TMP_DB"-wal "$TMP_DB"-shm' EXIT

# .backup makes a consistent snapshot. A plain copy of a live SQLite file can
# be torn if the dev server writes during the read.
echo "==> Snapshot the database"
sqlite3 "$LOCAL_DB" ".backup '$TMP_DB'"

# The VPS never needs the X session cookies. It does not refresh bookmarks, and
# the box is on the public internet.
echo "==> Remove the X cookies from the copy"
sqlite3 "$TMP_DB" "DELETE FROM Setting WHERE key IN ('x_auth_token','x_ct0');"
sqlite3 "$TMP_DB" "VACUUM;"

LEFT="$(sqlite3 "$TMP_DB" "SELECT count(*) FROM Setting WHERE key IN ('x_auth_token','x_ct0');")"
if [ "$LEFT" != "0" ]; then
  echo "error: cookies still present in the copy, stopping" >&2
  exit 1
fi

echo "==> Upload ($(du -h "$TMP_DB" | cut -f1))"
rsync -avz --progress "$TMP_DB" "$SSH_HOST:$REMOTE_DIR/prisma/dev.db.new"

# better-sqlite3 keeps the file open, so stop the service before the swap.
echo "==> Swap the file and restart"
ssh "$SSH_HOST" "
  set -e
  cd '$REMOTE_DIR'
  systemctl stop '$SERVICE'
  rm -f prisma/dev.db-wal prisma/dev.db-shm
  mv prisma/dev.db.new prisma/dev.db
  systemctl start '$SERVICE'
  sleep 2
  systemctl is-active '$SERVICE'
"

echo "==> Done. https://siftly.naveenreddy61.dev"
