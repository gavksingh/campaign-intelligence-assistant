# Deploying to Vercel

Step-by-step guide for deploying the Campaign Intelligence Assistant to Vercel.

## Prerequisites

- GitHub account with the repo pushed
- [Vercel](https://vercel.com) account (free tier works)
- [Neon](https://neon.tech) Postgres database (free tier works)
- Google Gemini API key ([get one here](https://ai.google.dev/gemini-api/docs/api-key))

## 1. Set Up Neon Postgres

Follow the [Neon setup guide](setup_neon.md) to create a database with pgvector enabled.

## 2. Import Project in Vercel

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your GitHub repository
3. Vercel will auto-detect the Next.js frontend in `web/` and the Python API in `api/`

## 3. Configure Environment Variables

In Vercel project **Settings → Environment Variables**, add:

| Variable | Value | Example |
|----------|-------|---------|
| `GOOGLE_API_KEY` | Your Gemini API key | `AIza...` |
| `DATABASE_URL` | Neon connection string (asyncpg format) | `postgresql+asyncpg://user:pass@ep-xxx.region.aws.neon.tech/campaign_intel?ssl=require` |
| `LLM_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-004` |

## 4. Deploy

Click **Deploy**. Vercel will:
- Build the Next.js frontend from `web/`
- Deploy the Python API as serverless functions from `api/`

## 5. Seed the Database

After deployment, seed your Neon database from your local machine:

```bash
DATABASE_URL="postgresql+asyncpg://user:pass@ep-xxx.region.aws.neon.tech/campaign_intel?ssl=require" \
GOOGLE_API_KEY="your-key" \
python -m data.seed
```

## 6. Verify

- Visit your Vercel URL — the Next.js chat UI should load
- Try the health endpoint: `https://your-app.vercel.app/api/health`
- Send a test chat message through the UI

## Troubleshooting

- **Function timeout**: Vercel free tier has a 10s function timeout. Upgrade to Pro for 60s if agent responses are slow.
- **Cold starts**: First request after idle may be slow. Subsequent requests are fast.
- **Database connection**: Ensure `?ssl=require` is in your DATABASE_URL for Neon.
- **Missing pgvector**: Run `CREATE EXTENSION IF NOT EXISTS vector;` on your Neon database.
