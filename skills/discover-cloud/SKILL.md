---
name: discover-cloud
description: Automatically discover cloud computing and serverless skills when working with cloud. Activates for cloud development tasks.
---

# Cloud Skills Discovery

Provides automatic access to comprehensive cloud skills.

## When This Skill Activates

This skill auto-activates when you're working with:
- cloud
- Modal
- serverless
- functions
- AWS
- GCP
- Azure
- deployment

## Hanzo Reality (read first)

Hanzo's own cloud is NOT AWS/GCP-shaped: it is ONE Go binary
(`hanzoai/cloud`) on zip with ZAP-only transport, embedded subsystems, and
three deploy modes (HIP-0106/0114/0116/0117).
Load `cat skills/hanzo-cloud-architecture/SKILL.md` first; use the vendor
skills below for customer/BYO-provider infrastructure.

## Available Skills

### Quick Reference

The Cloud category contains 13 skills:

1. **aws/** — API Gateway, databases, EC2, IAM, Lambda, networking, storage
2. **gcp/** — compute, databases, IAM, networking, serverless, storage
3. **cloud-kubernetes-deployment** — generic K8s (see `skills/cloud-kubernetes-deployment.md`)

### Load Full Category Details

For complete descriptions and workflows:

```bash
cat skills/cloud/INDEX.md
```

This loads the full Cloud category index with:
- Detailed skill descriptions
- Usage triggers for each skill
- Common workflow combinations
- Cross-references to related skills

### Load Specific Skills

Load individual skills as needed:

```bash

```

## Progressive Loading

This gateway skill enables progressive loading:
- **Level 1**: Gateway loads automatically (you're here now)
- **Level 2**: Load category INDEX.md for full overview
- **Level 3**: Load specific skills as needed

## Usage Instructions

1. **Auto-activation**: This skill loads automatically when Claude Code detects cloud work
2. **Browse skills**: Run `cat skills/cloud/INDEX.md` for full category overview
3. **Load specific skills**: Use bash commands above to load individual skills

---

**Next Steps**: Run `cat skills/cloud/INDEX.md` to see full category details.
