# Purge rules

Edit `purge_rules.json` and add exact words or phrases to its `terms` array. The scheduled pipeline checks new posts and every Hugging Face yearly partition. A matching post causes **all posts from that channel** to be removed, the channel record to be removed, and the channel to be added to the persistent Cloudflare blacklist.

Matching is case-insensitive and uses whole-token/whole-phrase boundaries. Whitespace inside a phrase may vary; punctuation around it is allowed. A configured term is not matched inside a larger word. For example, `termux` matches `Try TERMUX!`, but not `termuxer` or `mytermux`.

Example:

```json
{
  "terms": [
    "termux",
    "api key",
    "إعلان ممنوع"
  ]
}
```

Because the action removes an entire channel's history, add only terms that are distinctive enough to identify unwanted channels. Do not use short/common words unless you intend broad removal.
