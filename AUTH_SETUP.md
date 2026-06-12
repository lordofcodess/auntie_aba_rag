# Auth Setup — Supabase + Google OAuth + Anonymous Quota

This is the one-time configuration you need to do in **Supabase** and **Google
Cloud** before the auth flow works. The application code is already wired —
once these dashboards are configured, sign-in will work end-to-end.

---

## 1. Create a Supabase project (5 min, free tier)

1. Go to <https://supabase.com> and create an account.
2. Click **New project**. Give it a name (e.g. `nana-aba-ai`).
3. Set a strong database password (you won't need it for auth, but keep it).
4. Pick the closest region.
5. Wait ~2 min for the project to provision.

When it's ready, go to **Project Settings → API** and copy:

- **Project URL** (`https://<project-ref>.supabase.co`) → this becomes
  `VITE_SUPABASE_URL`
- **anon public key** (the `eyJ...` JWT) → this becomes
  `VITE_SUPABASE_ANON_KEY`
- **JWT Secret** (under "JWT Settings" — click "Reveal") → this becomes
  the backend's `SUPABASE_JWT_SECRET`

---

## 2. Enable Google OAuth in Supabase (10 min)

You'll create OAuth credentials in Google Cloud and paste them into Supabase.

### Google Cloud Console side

1. Go to <https://console.cloud.google.com>.
2. Create a new project (or pick existing).
3. **APIs & Services → OAuth consent screen**:
   - User type: **External**
   - App name: `Nana Aba AI`
   - User support email: your email
   - Developer contact: your email
   - Save and continue (you can leave scopes/test users alone)
4. **APIs & Services → Credentials → Create credentials → OAuth client ID**:
   - Application type: **Web application**
   - Name: `Nana Aba AI Web`
   - **Authorized redirect URIs**: add the Supabase callback URL
     `https://<project-ref>.supabase.co/auth/v1/callback`
     (Find this exact URL in Supabase → Authentication → Providers → Google)
   - Save. Copy the **Client ID** and **Client Secret**.

### Supabase side

1. Supabase Dashboard → **Authentication → Providers → Google**.
2. Toggle **Enable**.
3. Paste the Client ID and Client Secret from Google.
4. Save.

For local dev, also add this redirect URL in Supabase →
**Authentication → URL Configuration → Redirect URLs**:
- `http://localhost:5173/*`
- Your production deploy URL (e.g. `https://your-app.vercel.app/*`)

---

## 3. Environment variables

### Frontend (`ug_advisor_frontend/.env`)

Add these — restart `npm run dev` afterwards (Vite reads .env at boot):

```
VITE_SUPABASE_URL=https://<project-ref>.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...your-anon-key
```

### Backend (`auntie_aba_rag_initial`)

Modern Supabase projects sign JWTs with **asymmetric keys** (RS256/ES256) and
expose the public keys at a JWKS endpoint. The backend only needs the project
URL — no shared secret required.

When running uvicorn locally:

```bash
SUPABASE_URL="https://<project-ref>.supabase.co" \
GEMINI_API_KEY=... \
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

If your project still uses the legacy HS256 symmetric secret (Dashboard →
Project Settings → JWT Keys → Legacy JWT Secret), also pass
`SUPABASE_JWT_SECRET="<secret>"`. The verifier auto-picks based on the JWT's
`alg` header, so it's fine to set both during a migration.

Optional tuning:

| Env var | Default | What it does |
|---|---|---|
| `ANON_QUOTA` | `3` | Anonymous messages per IP before quota kicks in |
| `ANON_QUOTA_WINDOW_S` | `86400` (24h) | Rolling window for the quota |
| `SUPABASE_ISSUER` | derived from `SUPABASE_URL` | Expected `iss` claim; defaults to `<SUPABASE_URL>/auth/v1` |

### Modal deploy

Add the Supabase config as a Modal secret:

```bash
modal secret create supabase-auth \
  SUPABASE_URL=https://<project-ref>.supabase.co
# (add SUPABASE_JWT_SECRET=<secret> too only if you still use the legacy HS256 flow)
```

Then add it to the `secrets=[...]` list of `fastapi_app` in `modal_app.py`:

```python
auth_secret = modal.Secret.from_name("supabase-auth")
# ...
@app.function(..., secrets=[gemini_secret, auth_secret])
```

(I haven't edited `modal_app.py` for this yet — do that before your next
`modal deploy`.)

---

## 4. Test the flow

With the backend running and the frontend dev server up:

1. Open `http://localhost:5173`. You should see a **Sign in** button in the
   sidebar (and no auth badge).
2. Send `1`, `2`, `3` chat messages anonymously. They should work.
3. Send a 4th message — the backend returns **HTTP 402** and the **LoginCard**
   modal opens with the "You've used your 3 free messages" prompt.
4. Click **Continue with Google** → Google consent screen → redirected back to
   the app, now signed in. Your email shows in the sidebar.
5. Send more messages — they all work; the anonymous quota no longer applies.

To reset the anonymous counter while testing (it's per-IP, in-memory):

- Restart uvicorn, OR
- Wait 24h, OR
- Use a different network / VPN

---

## 5. Files involved

**Backend**:
- [`auth.py`](auth.py) — JWT verification + anonymous IP quota
- [`api.py`](api.py) — `gate` dependency applied to `/chat`, `/voice/chat`,
  `/document/analyze`, `/cv/analyze`, `/transcript/analyze`; new
  `/auth/quota` and `/auth/me` endpoints

**Frontend** (`ug_advisor_frontend/src/`):
- `supabaseClient.ts` — initializes the Supabase JS client
- `AuthContext.tsx` — React context with `useAuth()` hook
- `LoginCard.tsx` — modal with the Google sign-in button
- `api.ts` — `setAuthToken()` keeps the JWT in scope; `handleAuthFailure`
  routes 402 responses to the `QuotaExceededError`
- `App.tsx` — sidebar auth badge, LoginCard mount, 402 handler bridge
- `main.tsx` — wraps the app in `<AuthProvider>`

---

## 6. What's not yet built (Phase 3)

- **Per-user quotas**: signed-in users have *no* extra rate-limit yet. You'll
  want this later (e.g. 100 messages/day). Store counters keyed by `user.id`
  instead of IP.
- **Email/password fallback**: only Google OAuth right now. If you need
  email/password, enable Email provider in Supabase and add a second button
  in `LoginCard`.
- **Email allowlist** (e.g. only `@st.ug.edu.gh`): can be enforced either in
  Supabase RLS or in `auth.py`'s `_verify_token` by inspecting `email`.
