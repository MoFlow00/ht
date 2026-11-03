# Tele-API V2

Keep the Cloudflare KV binding named `DATA_STORE`. Configure secret `ADMIN_API_KEY`. Optional variable `ALLOWED_ORIGIN` accepts comma-separated exact web origins.

## Retained keys
- `latest_posts`
- `blacklist`
- `curated_explorer_data`

New key: `purge_audit`, retaining the latest 200 admin actions.

## Endpoints
- `GET /api/health`
- `GET /api/feed?page=1&page_size=20&q=&lang=&hashtag=&link_type=`
- `GET /api/explorer`
- `GET /api/stats`
- `GET /api/verify_admin` with `X-Admin-Key`
- `POST /api/purge` with JSON `{"username":"channel"}`
- `POST /api/restore` with JSON `{"username":"channel"}`
- `GET /api/blacklist` with `X-Admin-Key`
- `GET /api/purge_audit` with `X-Admin-Key`

The old `/api/purge?username=...` format remains temporarily supported for the current frontend.
Restore removes a blacklist entry. It does not reconstruct historical Parquet rows already deleted.
KV has no multi-key transactions or atomic read-modify-write. The purge order writes the blacklist first, then the live feed. Keep crawler runs serialized. For stronger concurrent writes, migrate mutable administrative state to D1 or Durable Objects.
KV-based rate limiting is best-effort. Use Cloudflare Rate Limiting Rules for enforceable edge limits.
