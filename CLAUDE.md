# Siftly

Self-hosted Twitter/X bookmark manager with AI-powered categorization, search, and visualization.

## Quick Setup

```bash
./start.sh            # installs deps, sets up DB, opens browser
```

Or manually:

```bash
npm install
npx prisma generate && npx prisma db push
npx next dev
```

App runs at **http://localhost:3000**

## AI Provider — Gemini

Siftly uses the Google Gemini API for categorization, semantic tags, image
analysis, and search. It does not use Anthropic or OpenAI.

- `lib/gemini-client.ts` holds the client and the model name.
- The model name is the floating alias `gemini-flash-lite-latest`. The alias
  always points to the current model, so a new model needs no migration.
- `app/api/settings/route.ts` holds `ALLOWED_GEMINI_MODELS`. A model that is
  absent from this list gives status 400 on a save.
- The key comes from the `geminiApiKey` database row first. Then it comes from
  the `GOOGLE_GENERATIVE_AI_API_KEY` or `GEMINI_API_KEY` environment variable.

To do a test of the key and the model:

```bash
curl -X POST http://localhost:3000/api/settings/test \
  -H 'Content-Type: application/json' -d '{"provider":"gemini"}'
```

The response `{"working":true}` shows success.

## Key Commands

```bash
npx next dev          # Start dev server (port 3000)
npm run dev           # Start dev server (port 4000)
npx tsc --noEmit      # Type check
npx prisma studio     # Database GUI
npx prisma db push    # Apply schema changes to DB
npm run build         # Production build (type-checks; next dev does not)
```

## Project Structure

```
app/
  api/
    categorize/       # AI pipeline — POST start, GET status by polling, DELETE stop
    import/           # Bookmark JSON import + dedup
      live/           # X cookie credentials + POST sync — the refresh path
      x-oauth/        # X OAuth flow — present but unused
    search/ai/        # FTS5 + Gemini semantic search
    settings/
      test/           # POST — validates the Gemini key and model
    analyze/images/   # Vision analysis progress + trigger
    bookmarks/        # CRUD + filtering
    categories/       # Category management
    mindmap/          # Graph data
    stats/            # Dashboard counts
  import/             # 3-step import UI
  mindmap/            # Interactive force graph
  settings/           # API keys, model selection
  ai-search/          # Natural language search UI
  bookmarks/          # Browse + filter UI
  categorize/         # Pipeline monitor

lib/
  gemini-client.ts    # Gemini client, model alias, API key resolution
  x-sync.ts           # X bookmark sync + optional scheduler
  twitter-api.ts      # X API calls with the auth_token and ct0 cookies
  categorizer.ts      # AI categorization + default categories
  vision-analyzer.ts  # Image vision + semantic tagging
  fts.ts              # SQLite FTS5 full-text search
  rawjson-extractor.ts # Entity extraction from tweet JSON
  parser.ts           # Multi-format bookmark JSON parser
  exporter.ts         # CSV / JSON / ZIP export

prisma/schema.prisma  # SQLite schema (Bookmark, Category, MediaItem, Setting, ImportJob)
```

## Tech Stack

- **Next.js 16** (App Router, TypeScript)
- **Prisma 7** + **SQLite** (local, zero setup, FTS5 built in)
- **@google/generative-ai** — vision, tagging, categorization, search
- **@xyflow/react** — mindmap graph
- **Tailwind CSS v4**

## Environment Variables

See `.env.example` for the full list. Only `DATABASE_URL` is required (defaults to `file:./prisma/dev.db`).

## CLI for AI Agents

`cli/siftly.ts` provides direct database access without the Next.js server. Outputs JSON (pretty-printed on TTY, compact when piped). Must run from project root.

```bash
npx tsx cli/siftly.ts stats                          # Library statistics
npx tsx cli/siftly.ts categories                     # Categories with counts
npx tsx cli/siftly.ts search "AI agents"             # FTS5 keyword search
npx tsx cli/siftly.ts list --limit 5                 # Recent bookmarks
npx tsx cli/siftly.ts list --source like --category ai-resources --sort oldest
npx tsx cli/siftly.ts show <id|tweetId>              # Full bookmark detail
npm run siftly -- stats                              # Alternative via npm script
```

## Common Tasks

| Task | How |
|------|-----|
| Refresh bookmarks and update the VPS | See `docs/refresh-workflow.md` |
| Run AI pipeline | `POST /api/categorize` with `{}` body. Read `GET /api/categorize` for progress. |
| Add category | Edit `DEFAULT_CATEGORIES` in `lib/categorizer.ts` — description is passed verbatim to Gemini |
| Add known tool | Append domain to `KNOWN_TOOL_DOMAINS` in `lib/rawjson-extractor.ts` |
| Test API auth | `POST /api/settings/test` with `{"provider":"gemini"}` |
| Change the model | Edit `ALLOWED_GEMINI_MODELS` in `app/api/settings/route.ts` first, or a save gives status 400 |

## Deployment

The portal runs at `https://siftly.naveenreddy61.dev`. Basic authentication in
nginx guards every route. `docs/refresh-workflow.md` holds the full procedure.

- The local machine is the only writer. It holds the X cookies and runs the
  pipeline.
- The VPS holds a read copy. `scripts/push-to-vps.sh` sends the database and
  removes the X cookies from the copy.
- Each push replaces the whole file. A category change or a settings change made
  on the VPS is lost.
- The VPS files are in `/root/projects/siftly`. The service name is `siftly`. The
  app listens on `127.0.0.1:8002`.
- Use the Secure Shell alias `vps-rsync` for a command. The alias `vps` starts a
  tmux session, so it cannot take a command.
- `deploy/` holds the nginx site file and the systemd unit file.

## Database

SQLite at `prisma/dev.db`. After schema changes: `npx prisma db push`

Models: `Bookmark`, `MediaItem`, `BookmarkCategory`, `Category`, `Setting`, `ImportJob` — see `prisma/schema.prisma` for details.