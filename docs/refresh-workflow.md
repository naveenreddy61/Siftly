# How to Refresh the Bookmarks and Update the VPS

This guide gives the full procedure for a periodic refresh. Do this procedure
every few months, or when you want the new bookmarks on the portal.

The local machine is the only writer. It holds the X cookies and it runs the
artificial intelligence pipeline. The VPS holds a read copy for browsing.

## When to Do This Procedure

Do this procedure when one of these conditions is true:

- You added new bookmarks on X and you want them on the portal.
- The portal shows old data.
- The value of `x_last_sync` is more than one month old.

## Before You Start

You need three things:

1. Access to the X account in a web browser.
2. A terminal on the local machine, in the project root directory.
3. Secure Shell access to the VPS. The command `ssh vps` must work.

## Stage 1 — Start the Local Server

1. Open a terminal in the project root directory.
2. Start the server:

   ```bash
   npx next dev
   ```

3. Wait for the message `Ready`.
4. Open `http://localhost:3000` in the browser.

The command `npx next dev` uses port 3000. The script `npm run dev` uses port
4000 instead. This guide uses port 3000 everywhere.

## Stage 2 — Get New X Cookies

The cookies expire. A refresh fails with old cookies, so always get new cookies
first.

CAUTION: The two values in this stage give full access to the X account. Do not
put them in a file that Git tracks. Do not send them to the VPS.

1. Sign in to `https://x.com` in Chrome.
2. Press `F12` to open the developer tools.
3. Select the **Application** tab.
4. In the left panel, expand **Cookies**. Select `https://x.com`.
5. Find the row `auth_token`. Copy its value. The value has 40 characters.
6. Find the row `ct0`. Copy its value. The value has approximately 160
   characters.

## Stage 3 — Save the Cookies

Use the user interface or the command line. Both methods write the same two
database rows.

### Method A — The User Interface

1. Open `http://localhost:3000/import`.
2. Select the **Live Import** tab.
3. Find the **X Session Cookies** panel.
4. Put the `auth_token` value in the first field.
5. Put the `ct0` value in the second field.
6. Select **Save Credentials**.

The panel then shows the text `Credentials saved`.

### Method B — The Command Line

Replace `<auth_token>` and `<ct0>` with the two values:

```bash
curl -X POST http://localhost:3000/api/import/live \
  -H 'Content-Type: application/json' \
  -d '{"authToken":"<auth_token>","ct0":"<ct0>"}'
```

To confirm that the database holds both values, do this command:

```bash
curl -s http://localhost:3000/api/import/live
```

The field `hasCredentials` must be `true`.

## Stage 4 — Refresh the Bookmarks

1. Start the sync:

   ```bash
   curl -X POST http://localhost:3000/api/import/live/sync
   ```

2. Wait for the response. A sync of 3300 bookmarks needs approximately 50
   seconds.
3. Read the two counts in the response. The field `imported` gives the number of
   new records. The field `skipped` gives the number of records that the database
   already holds.

The sync reads the newest bookmarks first. It stops after 50 pages. It does not
use the value of `x_last_sync`, so it reads records that the database already
holds. This behaviour is correct. The `skipped` count shows those records.

WARNING: Do not send the sync command a second time after a timeout. The server
permits one sync at a time and answers with status 409. The first sync continues.
To see the result, read the bookmark count in the database.

## Stage 5 — Categorize the New Bookmarks

The pipeline gives categories, semantic tags, and image tags to each new record.
It processes only the records that have no data, so a second run is cheap.

1. Start the pipeline:

   ```bash
   curl -X POST http://localhost:3000/api/categorize \
     -H 'Content-Type: application/json' -d '{}'
   ```

2. Read the field `total` in the response. This is the number of records in the
   run.
3. Read the progress at intervals:

   ```bash
   curl -s http://localhost:3000/api/categorize
   ```

4. Wait for the value `idle` in the field `status`. A run of 517 records needs
   approximately 13 minutes.

WARNING: Do not change a file in `app/` or in `lib/` during a run. The server
keeps the progress in memory. A file change makes the server compile the code
again, and the run stops.

To confirm that no record remains, do this command:

```bash
sqlite3 prisma/dev.db "SELECT count(*) FROM Bookmark b WHERE NOT EXISTS (SELECT 1 FROM BookmarkCategory bc WHERE bc.bookmarkId=b.id);"
```

The result must be `0`.

## Stage 6 — Push the Database to the VPS

The script does all the work. It makes a consistent snapshot of the database. It
removes the two X cookies from the copy. It sends the copy to the VPS. It stops
the service, replaces the file, and starts the service again.

1. Do the script from the project root directory:

   ```bash
   ./scripts/push-to-vps.sh
   ```

2. Wait for the line `Done`. A transfer of 62 megabytes needs approximately 10
   seconds.

## Stage 7 — Confirm the Result

1. Confirm that the portal refuses an anonymous request:

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' https://siftly.naveenreddy61.dev/bookmarks
   ```

   The result must be `401`.

2. Confirm the record count. Replace `<password>` with the portal password:

   ```bash
   curl -s -u naveen:<password> https://siftly.naveenreddy61.dev/api/stats
   ```

   The field `totalBookmarks` must equal the local count. The field
   `uncategorizedCount` must be `0`.

3. Open `https://siftly.naveenreddy61.dev` in the browser and sign in.

## How to Update the Code on the VPS

The script `push-to-vps.sh` sends the database only. It does not send code.
Therefore a change to a page, a component, or a route needs this separate
procedure.

Do this procedure only when you changed the code. A data refresh does not need
it.

1. Commit the change on the local machine and push it to `origin/main`.
2. Do a type check first. The command `next dev` does not check types, but
   `npm run build` does:

   ```bash
   npx tsc --noEmit
   ```

3. If step 2 gives an error, correct the error. A type error stops the build on
   the VPS.
4. Update the code on the VPS and start the service again:

   ```bash
   ssh vps-rsync 'set -e; cd /root/projects/siftly; git fetch origin -q; git reset --hard origin/main -q; npm ci; npx prisma generate; npm run build; systemctl restart siftly'
   ```

5. Confirm that the service is active:

   ```bash
   ssh vps-rsync 'systemctl is-active siftly'
   ```

   The result must be `active`.

WARNING: Do not copy `node_modules` from the local machine to the VPS. The local
machine uses a macOS binary. The VPS needs a Linux binary. The command `npm ci`
on the VPS makes the correct binary.

## Important Limits

The VPS is for browsing only. Each push replaces the whole database file.
Therefore the VPS loses these changes:

- A category change that you make on the VPS.
- A settings change that you make on the VPS. This includes the model name and
  the application programming interface key.

Make all such changes on the local machine. Then push the database again.

## Troubleshooting

### The sync gives an error about the credentials

The response holds the text `X credentials not configured`. The database has no
cookies. Do Stage 3 again.

### The sync gives an error about the response format

The response holds the text `Twitter API response format has changed`. Two
causes are possible:

1. The cookies expired. This is the common cause. Do Stage 2 and Stage 3 again.
2. X changed the API. In this case, the file `lib/twitter-api.ts` needs a
   change.

### The sync answers with status 409

A sync is already active. Wait for it to finish. Do not start a second sync.

### The pipeline stops before the end

Read the field `lastError` in the progress response. If the field names the
application programming interface key, open the settings page and check the key.
Start the pipeline again. It continues from the first record that has no data.

### The portal gives status 502

The service on the VPS is not active. Read the state and the logs:

```bash
ssh vps-rsync 'systemctl is-active siftly; journalctl -u siftly -n 30 --no-pager'
```

To start the service again:

```bash
ssh vps-rsync 'systemctl restart siftly'
```

### The portal shows old data

The push did not finish. Do Stage 6 again. Then read the file date on the VPS:

```bash
ssh vps-rsync 'ls -l /root/projects/siftly/prisma/dev.db'
```

## Reference

| Item | Value |
| ------ | ------- |
| Local address | `http://localhost:3000` |
| Local database | `prisma/dev.db` |
| Portal address | `https://siftly.naveenreddy61.dev` |
| Portal user | `naveen` |
| VPS host alias for a command | `vps-rsync` |
| VPS host alias for a terminal | `vps` |
| VPS directory | `/root/projects/siftly` |
| VPS service | `siftly` |
| VPS port | `127.0.0.1:8002` |
| Sync script | `scripts/push-to-vps.sh` |

The alias `vps` starts a tmux session, so it cannot take a command. Use
`vps-rsync` for every command.

The portal password is not in this file. The VPS holds the hash in
`/etc/nginx/.htpasswd-siftly`. To set a new password, do this command from a
terminal:

```bash
ssh -t vps-rsync htpasswd -B /etc/nginx/.htpasswd-siftly naveen
```

WARNING: The command `htpasswd` needs a real terminal. Without a terminal it
accepts an empty password and gives no error. An empty password gives full access
to the portal. Do not run this command through the `!` prefix in Claude Code,
because that is not a terminal. Use a terminal window.

After the command, do a test of the result:

```bash
ssh vps-rsync 'htpasswd -vb /etc/nginx/.htpasswd-siftly naveen ""'
```

The result must be `password verification failed`.
