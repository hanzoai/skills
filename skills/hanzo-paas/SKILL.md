---
name: hanzo-paas
description: "Deploy and manage applications on Hanzo PaaS. Manage projects, environments, deployments, domains, logs, and cloud services via the hanzo-tools-paas MCP tool or REST API."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "hanzo-tools-paas",
              "label": "Install Hanzo PaaS Tools (pip)",
            },
          ],
      },
  }
---

# Hanzo PaaS — Platform as a Service

`pip install hanzo-tools-paas`

Deploy and manage applications on Hanzo PaaS (platform.hanzo.ai). Supports Dokploy-based hosting, Docker/Compose deployments, domain management, and IAM integration.

## Authentication

```bash
# Login via Hanzo CLI (stores token at ~/.hanzo/auth/token.json)
hanzo login

# Or set environment variables
export HANZO_IAM_URL=https://hanzo.id
export HANZO_IAM_CLIENT_ID=your-client-id
export HANZO_IAM_CLIENT_SECRET=your-client-secret
```

## MCP Tool (Recommended)

The `paas` MCP tool provides a unified interface via `hanzo-mcp`:

```python
# Via hanzo-mcp server — single tool with action parameter
# Tool: paas(action, org, project, environment, container)
```

### IAM Actions

```python
# Who am I?
paas(action="whoami")
# → {"sub": "user-id", "name": "...", "email": "...", "organization": "hanzo"}

# List users
paas(action="users")
# → {"count": 5, "users": [{"id": "...", "name": "...", "email": "..."}]}

# List organizations
paas(action="orgs")
# → {"count": 3, "organizations": [...]}

# List roles
paas(action="roles")
# → {"count": 4, "roles": [...]}
```

### Deployment Actions

```python
# List projects in org
paas(action="projects", org="hanzo")
# → {"org": "hanzo", "count": 12, "projects": [...]}

# List environments for a project
paas(action="env", org="hanzo", project="my-project")
# → {"environments": [{"id": "...", "name": "production"}, ...]}

# List containers/deployments
paas(action="deployments", org="hanzo", project="my-project", environment="prod")
# → {"containers": [{"id": "...", "name": "web", "image": "...", "status": "running", "replicas": 3}]}

# Get deployment details
paas(action="deploy", org="hanzo", project="my-project", environment="prod", container="web")

# View container logs
paas(action="logs", org="hanzo", project="my-project", environment="prod", container="web")

# Redeploy (rolling restart)
paas(action="redeploy", org="hanzo", project="my-project", environment="prod", container="web")
```

### Cloud Services

```python
# List managed services and cluster info
paas(action="services")
# → {"cluster": {...}, "templates": {...}}
```

## REST API (Direct)

Base URL: `https://platform.hanzo.ai/api`

```bash
TOKEN=$(cat ~/.hanzo/auth/token.json | jq -r .access_token)

# List projects
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://platform.hanzo.ai/api/project.all" | jq

# Create project
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "description": "My application"}' \
  "https://platform.hanzo.ai/api/project.create" | jq

# Deploy application
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"applicationId": "<app-id>"}' \
  "https://platform.hanzo.ai/api/application.deploy" | jq

# Read logs
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://platform.hanzo.ai/api/application.readLogs?applicationId=<id>" | jq
```

## Docker Compose Deployments

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "projectId": "<project-id>",
    "name": "my-stack",
    "composeFile": "services:\n  web:\n    image: nginx"
  }' "https://platform.hanzo.ai/api/compose.create" | jq
```

## Domain Management

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "applicationId": "<app-id>",
    "host": "myapp.hanzo.ai",
    "certificateType": "letsencrypt"
  }' "https://platform.hanzo.ai/api/domain.create" | jq
```

## Environment Variables

```bash
HANZO_IAM_URL=https://hanzo.id           # IAM endpoint
HANZO_IAM_CLIENT_ID=...                   # OAuth client ID
HANZO_IAM_CLIENT_SECRET=...               # OAuth client secret
HANZO_PLATFORM_URL=https://platform.hanzo.ai  # PaaS endpoint
```

## Key Endpoints

| Endpoint                      | Purpose                |
| ----------------------------- | ---------------------- |
| `project.all`                 | List all projects      |
| `project.create`              | Create project         |
| `application.create`          | Create app service     |
| `application.deploy`          | Trigger deployment     |
| `application.readLogs`        | Read app logs          |
| `compose.create`              | Create compose service |
| `compose.deploy`              | Deploy compose         |
| `domain.create`               | Add domain             |
| `application.saveEnvironment` | Set env vars           |
