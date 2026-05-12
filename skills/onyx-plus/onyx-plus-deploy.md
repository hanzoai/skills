---
name: onyx-plus-deploy
description:  deploy pipeline -- Cloud Build (no GitHub builders), manifest bump, kubectl set image, strict dev→test→main
---

#  Deploy Pipeline

**Category**: 
**Related Skills**: `onyx-plus/onyx-plus.md`, `onyx-plus/onyx-plus-onyxd.md`, `onyx-plus/onyx-plus-internal.md`

## Build path -- Cloud Build only

 images are built via GCP Cloud Build, NOT GitHub Actions. Per the workspace CLAUDE.md: "NO GITHUB BUILDERS." Each repo carries a `cloudbuild.yaml`:

```bash
# Submit a build for the daemon
cd ~/work//onyxd
gcloud builds submit --config=cloudbuild.yaml \
    --project=-registry \
    --gcs-source-staging-dir=gs://-registry_cloudbuild/source
```

Output: `us-docker.pkg.dev/-registry/onyx-plus/{name}:{semver}`.

The `VERSION` file is the immutable tag. Cloud Build refuses to push when the tag already exists in GAR -- so each release bumps `VERSION` before submission.

## Repos with cloudbuild + VERSION

| Repo | Image | Current tag |
|---|---|---|
| `onyx-plus/onyxd` | `us-docker.pkg.dev/-registry/onyx-plus/onyxd` | `0.2.2` |
| `onyx-plus/admin` | `us-docker.pkg.dev/-registry/onyx-plus/admin` | `1.0.0` |
| `onyx-plus/internal` | `us-docker.pkg.dev/-registry/onyx-plus/internal` | `0.1.1` |
| `onyx-plus/verify` | (TBD -- not yet cloudbuilt) | -- |
| `onyx-plus/onboarding` | (TBD -- not yet cloudbuilt) | -- |

## Release procedure (strict)

1. **Bump `VERSION`** in the repo, e.g. `0.2.1` → `0.2.2` (patch-only by default; major bumps prohibited per CLAUDE.md without explicit instruction).
2. **Commit + tag**:
   ```bash
   git commit VERSION -m "release: v0.2.2"
   git tag v0.2.2
   git push origin main v0.2.2
   ```
3. **Submit Cloud Build**:
   ```bash
   gcloud builds submit --config=cloudbuild.yaml \
       --project=-registry \
       --gcs-source-staging-dir=gs://-registry_cloudbuild/source
   ```
   Build takes ~3 min for the Go daemon, ~5-7 min for Next.js (internal) due to vendor/pnpm.
4. **Verify image landed**:
   ```bash
   gcloud artifacts docker images list \
       us-docker.pkg.dev/-registry/onyx-plus/onyxd \
       --include-tags --filter='tags~0.2.2'
   ```
5. **Bump K8s manifest tag** in `~/work//admin/universe/k8s/{dev,test,main}/services.yaml`, commit, push.
6. **Roll out dev**, smoke, **promote test**, smoke, **promote main**, smoke -- in that order. Strict per CLAUDE.md ("Do NOT bump all 3 envs in parallel").

## Deploy commands

```bash
# Dev
gcloud container clusters get-credentials dev \
    --region=us-central1 --project=-devnet
kubectl -n  set image statefulset/onyxd \
    onyxd=us-docker.pkg.dev/-registry/onyx-plus/onyxd:0.2.2
kubectl -n  rollout status statefulset/onyxd --timeout=180s

# Smoke
kubectl -n  exec onyxd-0 -- wget -qO- localhost:8090/v1//healthz
# Expect: {"service":"onyxd","status":"ok"}

# Promote test
gcloud container clusters get-credentials test \
    --region=us-central1 --project=-testnet
kubectl -n  set image statefulset/onyxd \
    onyxd=us-docker.pkg.dev/-registry/onyx-plus/onyxd:0.2.2
kubectl -n  rollout status statefulset/onyxd --timeout=180s
# Smoke

# Promote main
gcloud container clusters get-credentials main \
    --region=us-central1 --project=-mainnet
kubectl -n  set image statefulset/onyxd \
    onyxd=us-docker.pkg.dev/-registry/onyx-plus/onyxd:0.2.2
kubectl -n  rollout status statefulset/onyxd --timeout=180s
# Smoke
```

Same pattern for `deployment/admin` and `deployment/internal` (just `set image deployment/...` instead of `statefulset/...`).

## GCP topology

| Project | Env | Cluster | Region |
|---|---|---|---|
| `-devnet` | dev | `dev` | `us-central1` |
| `-testnet` | test | `test` | `us-central1` |
| `-mainnet` | main | `main` | `us-central1` |
| `-registry` | (registry only) | -- | `us` (multi-region) |

Strict separation from `liquidity-registry` and `ghcr.io/hanzoai`. Per CLAUDE.md container registry rule.

## DNS

| Host | Resolves to | Notes |
|---|---|---|
| `.{env}.` | `34.136.247.167` / `35.188.49.92` / `34.57.235.223` | Operator dashboard (admin) -- upstream LB, not direct to Traefik |
| `-api.{env}.` | same as above | API (onyxd) |
| `internal..{env}.` | same as above | Internal docs (internal) |
| `verify.{env}.` | TBD | Standalone IDV |
| `onboarding.{env}.` | TBD | BD/KYC onboarding |

There's an upstream gateway in front of each cluster -- the cluster's own Traefik LB IPs (`34.61.160.79` dev, etc.) are different from the DNS-resolved IPs. New vhosts may need the upstream gateway config updated as well as the cluster Ingress.

## Pre-existing onyxd manifest issue

`admin/universe/k8s/{env}/services.yaml` references `volumeMounts[].name: data` and `name: tmp` without declaring matching `volumes:` entries. `kubectl apply -f services.yaml` rejects the StatefulSet update each time:

```
* spec.template.spec.containers[0].volumeMounts[0].name: Not found: "data"
* spec.template.spec.containers[0].volumeMounts[1].name: Not found: "tmp"
```

Workaround used today: `kubectl set image statefulset/onyxd ...` -- only updates the container image, doesn't re-apply the whole spec. Worth fixing in a follow-up so the manifest is applyable end-to-end.

## Internal docs site specifics

- `internal/Dockerfile` uses pnpm + a vendored `vendor/hanzo-docs/packages/` workspace (the published `@hanzo/docs` package has unresolved `workspace:*` deps including `@hanzo/docs-tailwind` which is unpublished).
- A `.pnpmfile.cjs` hook strips `workspace:*` refs to packages not present in the local vendor dir.
- `internal-secrets` K8s Secret in each cluster carries `AUTH_SECRET` (real) + `AUTH_GOOGLE_ID` / `AUTH_GOOGLE_SECRET` (placeholders -- replace with a real Google OAuth client before sign-in works).
- Redirect URIs: `https://internal..{env}./api/auth/callback/google`.

## Source pointers

| Subject | Source |
|---|---|
| onyxd cloudbuild | `~/work//onyxd/cloudbuild.yaml` |
| admin cloudbuild | `~/work//admin/cloudbuild.yaml` |
| internal cloudbuild | `~/work//internal/cloudbuild.yaml` |
| K8s manifests | `~/work//admin/universe/k8s/{dev,test,main}/` |

## Related skills

- `onyx-plus/onyx-plus.md` - umbrella
- `onyx-plus/onyx-plus-onyxd.md` - daemon details
- `onyx-plus/onyx-plus-internal.md` - internal docs site specifics
- `hanzo/cloud-kubernetes-deployment.md` - general K8s deployment

---

**Last Updated**: 2026-05-12
**Category**: 
**Related**: deploy, cloud-build, gke, kubectl, semver, gar
