# Setting Up Neon Postgres (pgvector)

This guide walks through setting up a Neon Postgres database with pgvector for the Campaign Intelligence Assistant.

## 1. Create a Neon Account

Sign up at [neon.tech](https://neon.tech) (free tier available).

## 2. Create a New Project

1. Click **New Project** in the Neon dashboard
2. Name: `campaign-intelligence` (or your preference)
3. Region: Choose closest to your Vercel deployment region
4. Postgres version: 16+

## 3. Enable pgvector Extension

Connect to your database and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Verify:

```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## 4. Get Connection String

In the Neon dashboard, go to **Connection Details** and copy the connection string. It will look like:

```
postgresql://user:password@ep-xxx.region.aws.neon.tech/campaign_intel?sslmode=require
```

For the async driver used by this app, modify it to use `asyncpg`:

```
postgresql+asyncpg://user:password@ep-xxx.region.aws.neon.tech/campaign_intel?ssl=require
```

## 5. Add to Vercel Environment Variables

In your Vercel project settings, add:

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | `postgresql+asyncpg://user:password@ep-xxx.region.aws.neon.tech/campaign_intel?ssl=require` |
| `GOOGLE_API_KEY` | Your Google Gemini API key |

## 6. Seed the Database

After deploying, seed from your local machine:

```bash
DATABASE_URL="postgresql+asyncpg://user:password@ep-xxx.region.aws.neon.tech/campaign_intel?ssl=require" \
python -m data.seed
```

## 7. Verify

```sql
SELECT COUNT(*) FROM campaigns;
SELECT COUNT(*) FROM campaign_embeddings;
```

Both should return 18 (the number of mock campaigns).
