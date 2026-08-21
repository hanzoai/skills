# Deployment Skills

## Category Overview

**Total Skills**: 6
**Category**: deployment

## Hanzo Reality (read first)

The skills below cover third-party platforms (Heroku, Netlify). **Hanzo
services do NOT deploy this way.** Hanzo deployment is HIP-0117: ONE binary,
three modes — `cloud serve` (no Kubernetes), `cloud cluster init` (fetch k3s
+ operator + `services.hanzo.ai` CRs), or `helm install` (BYO k8s). CI/CD is
Gitea Actions on git.hanzo.ai; GitHub is the OSS mirror.

Load: `cat skills/hanzo-cloud-architecture/SKILL.md`

## Skills in This Category

### heroku-addons.md
**Description**: Adding databases (Postgres, MySQL, MongoDB) to Heroku apps

**Load this skill**:
```bash
cat skills/deployment/heroku-addons.md
```

---

### heroku-deployment.md
**Description**: Deploying web applications, APIs, or worker processes to Heroku

**Load this skill**:
```bash
cat skills/deployment/heroku-deployment.md
```

---

### heroku-troubleshooting.md
**Description**: Application crashes on Heroku with H10, H12, H13, H14 errors

**Load this skill**:
```bash
cat skills/deployment/heroku-troubleshooting.md
```

---

### netlify-deployment.md
**Description**: Deploying static sites, JAMstack apps, or frontend frameworks to Netlify

**Load this skill**:
```bash
cat skills/deployment/netlify-deployment.md
```

---

### netlify-functions.md
**Description**: Building API endpoints for JAMstack applications

**Load this skill**:
```bash
cat skills/deployment/netlify-functions.md
```

---

### netlify-optimization.md
**Description**: Improving site load times and Core Web Vitals

**Load this skill**:
```bash
cat skills/deployment/netlify-optimization.md
```

---

## Loading All Skills

```bash
# List all skills in this category
ls skills/deployment/*.md

# Load specific skills
cat skills/deployment/heroku-addons.md
cat skills/deployment/heroku-deployment.md
cat skills/deployment/heroku-troubleshooting.md
# ... and 3 more
```

## Related Categories

See `skills/README.md` for the complete catalog of all categories and gateway skills.

---

**Browse**: This index provides a quick reference. Load the `deployment` gateway skill for common workflows and integration patterns.

```bash
cat skills/deployment/SKILL.md
```
