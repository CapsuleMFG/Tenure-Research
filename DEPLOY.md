# Deploy Tenure Brief to Streamlit Community Cloud

End-to-end checklist for the first deploy. After the first deploy, the
weekly + daily GitHub Actions keep the data fresh automatically.

## 1. Prerequisites

- The repo is pushed to GitHub (it is: `github.com/CapsuleMFG/Tenure-Research`).
- You have a GitHub account that can authorize Streamlit Cloud.
- `data/clean/observations.parquet` and `data/clean/forecasts.json` are
  committed to `main` (no longer gitignored as of the deploy-prep pass).

## 2. Initial deploy

1. Sign in to **[share.streamlit.io](https://share.streamlit.io)** with your
   GitHub account. Authorize Streamlit to read the repo.
2. Click **New app**.
3. Settings:
   - **Repository:** `CapsuleMFG/Tenure-Research`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL** *(optional)*: pick a slug like `tenure-brief`.
4. Click **Deploy**. First build takes 5–10 minutes (installs Python deps,
   builds the wheel).
5. When the build finishes, Streamlit hands you a public URL like
   `https://tenure-brief.streamlit.app`. Open it.

## 3. Smoke test

Open the deployed URL and verify:

- [ ] **Home (Brief)** — hero + sample-scenario card + 4 step cards render
- [ ] **Plan** — region picker + tabs work; margin card computes
- [ ] **Costs** — daily futures cards render with real prices
- [ ] **Pricing** — share calculator math runs
- [ ] **Series**, **Forecast**, **Catalog**, **Methodology**, **About** all load
- [ ] **Mobile** — open the URL on a phone; verify columns stack and
      typography is readable

If any page fails, check the Streamlit Cloud logs (manage app → logs).
Common first-deploy issues: missing requirement (run `uv export` to
regenerate `requirements.txt`); cold-start timeouts (acceptable, second
load is fast).

## 4. Enable scheduled refreshes

The weekly and daily workflows are already in `.github/workflows/` but
won't run until GitHub Actions is enabled on your fork. To activate:

1. Go to the repo on GitHub → **Actions** tab.
2. If prompted, click **I understand my workflows, go ahead and enable them**.
3. Trigger the weekly workflow once manually to verify it works:
   - Actions → "Refresh data + rebake forecasts (weekly)" → **Run workflow**.
4. After the run finishes, check the commit history on `main`. You should
   see a commit like `data: weekly refresh 2026-MM-DD` from
   `tenure-brief-bot`. Streamlit Cloud auto-redeploys on push.

The weekly cron is set for **Sundays 21:00 UTC**; the daily cron is set
for **22:00 UTC every day**. The daily one only does the cheap futures
refresh — full re-bake is weekly.

## 5. After deploy

### Sharing

- **Slack / Twitter / iMessage** — the OG meta tags are wired so the
  preview card renders cleanly. Test by sharing the URL into Slack.
- **Bookmarkable scenarios** — try the workflow yourself: pick a region,
  adjust inputs, hit Copy shareable link, bookmark the resulting URL.

### Admin

The admin panel (Refresh data / Rebuild forecast cache) is gated by
`?admin=1` in the URL. Visit
`https://tenure-brief.streamlit.app/?admin=1` to expose it. It's hidden
from regular visitors so they don't accidentally trigger a multi-minute
refresh.

### v2.0 commodity tools

The original v2.0 Decide + Feedlot Breakeven tools are still in the
codebase and accessible via `?advanced=1` (or the same `?admin=1` flag).
They're hidden from the primary sidebar nav so the v3.0 direct-market
audience doesn't see noise, but a feedlot operator who knows about them
can still get there.

## 6. Cost

- Streamlit Community Cloud: **free** for public apps with up to 1GB RAM
  per app. We're well under.
- GitHub Actions: **free** for public repos within the standard usage
  quota. The weekly + daily crons combined use a few CI-minutes per week.

## 7. Tearing down

If you decide to take it offline:

1. Streamlit Cloud: manage app → **Delete app**. URL stops serving immediately.
2. (Optional) Disable workflows: Actions → each workflow → ··· menu →
   **Disable workflow**. Stops the scheduled refresh.

The repo and history remain. Re-deploy is a 5-minute click.

## 8. What to watch first

After deploy:

- **Cold-start latency** — Streamlit Cloud sleeps inactive apps. First
  visit after a sleep takes 30–60 seconds to spin up. Repeat visits are fast.
- **Plotly bundle size** — adds ~3MB to first paint. Acceptable but
  noticeable on slow connections.
- **Mobile layout** — CSS rules are in place but untested on real hardware.
  First mobile visitor will tell you whether anything's broken.
- **First user feedback** — if you can, watch a real direct-market rancher
  navigate the Plan page for 5 minutes. That's the single highest-value
  signal you can get.

## 9. Out-of-scope (future v3.1+)

- Custom domain (`tenurebrief.com` → Streamlit URL via CNAME). Streamlit
  Cloud supports this on paid plans.
- Authentication / private mode. Free tier is public-only.
- User accounts + server-side scenario sync. Currently scenarios live
  in browser localStorage only.
- AMS LMR daily cash. Requires eAuth Level 2; deferred indefinitely.
- Real mobile device QA.
- Real direct-market rancher walkthrough.
