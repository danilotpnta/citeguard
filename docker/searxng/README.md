# SearXNG setup

Self-hosted meta-search engine used as a last-resort web search fallback in citeguard.

## First-time setup

**1. Create your settings file from the example:**

```bash
cp settings.yml.example settings.yml
```

**2. Generate a secret key and add it to `settings.yml`:**

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Open `settings.yml` and replace `your_secret_key_here` with the generated value.

> `settings.yml` is gitignored — it contains your secret key and should never be committed.

**3. Start the container:**

```bash
docker compose -f docker/searxng/docker-compose.yml up -d
```

**4. Add to your `.env`:**

```
SEARXNG_URL=http://localhost:8080
```

## Verify it's working

```bash
curl "http://localhost:8080/search?q=attention+is+all+you+need&format=json&categories=science" | python3 -m json.tool | head -30
```

You should see search results from Google Scholar, Semantic Scholar, arXiv, etc.