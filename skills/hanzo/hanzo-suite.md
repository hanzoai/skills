# Hanzo Suite - Business ERP/CRM Suite (Odoo 18 Fork)

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-commerce.md`, `hanzo/hanzo-platform.md`

## Overview

Hanzo Business Suite is a **fork of Odoo 18.0** (Community Edition) providing a full-featured open-source ERP/CRM platform tightly integrated with Hanzo AI infrastructure. It includes CRM, e-commerce, warehouse management, project management, billing/accounting, POS, HR, marketing, manufacturing, and 100+ built-in modules. Runs as a Python web application backed by PostgreSQL. Live at `team.hanzo.ai` and `hanzo.team`.

### Why Hanzo Suite?

- **Full ERP in one platform**: CRM, accounting, inventory, HR, e-commerce, POS, project management
- **Odoo 18.0 base**: Mature, battle-tested ERP framework with large addon ecosystem
- **AI integration**: Positioned for Hanzo AI enhancements on top of Odoo's module system
- **Self-hosted**: Docker Compose deployment with PostgreSQL, Traefik reverse proxy
- **Modular**: 100+ built-in addons, plus custom addons in `/mnt/extra-addons`

### Tech Stack

- **Language**: Python 3.10+ (core framework)
- **Web framework**: Werkzeug + custom Odoo HTTP layer
- **ORM**: Odoo ORM (custom, PostgreSQL-backed, declarative field definitions)
- **Database**: PostgreSQL 16
- **Frontend**: Odoo Web Client (JavaScript/OWL framework)
- **Templates**: Jinja2 + QWeb (Odoo's XML-based template engine)
- **Container**: `odoo:18` official Docker image
- **Reverse proxy**: Traefik (production)
- **License**: LGPL-3.0

### OSS Base

Repo: `hanzoai/suite`. Default branch: `main`. Fork of Odoo 18.0 Community Edition.

## When to use

- Running internal business operations (CRM, accounting, inventory, HR)
- E-commerce storefront with integrated warehouse and shipping
- Project management with time tracking and billing
- Point-of-sale for retail operations
- Custom business workflow automation via Odoo modules
- Financial reporting and accounting

## Hard requirements

1. **PostgreSQL 16** database
2. **Python 3.10+** with system dependencies (lxml, Pillow, psycopg2, etc.)
3. **Docker** (recommended deployment method)
4. **wkhtmltopdf** for PDF report generation (included in Odoo Docker image)

## Quick reference

| Item | Value |
|------|-------|
| Live URL | `https://team.hanzo.ai` / `https://hanzo.team` |
| Web port | 8069 (mapped to 10017 in dev compose) |
| Live chat port | 8072 (mapped to 20017 in dev compose) |
| Odoo version | 18.0 (final) |
| Python | 3.10+ |
| Database | PostgreSQL 16 |
| ORM | Odoo ORM (custom) |
| Default branch | `main` |
| License | LGPL-3.0 |
| Repo | `github.com/hanzoai/suite` |

## One-file quickstart

### Docker Compose (development)

```yaml
# compose.yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: odoo
      POSTGRES_PASSWORD: "${DB_PASSWORD}"
      POSTGRES_DB: postgres
    volumes:
      - ./postgresql:/var/lib/postgresql/data

  odoo:
    image: odoo:18
    depends_on:
      - db
    ports:
      - "8069:8069"
      - "8072:8072"
    environment:
      HOST: db
      USER: odoo
      PASSWORD: "${DB_PASSWORD}"
    volumes:
      - ./addons:/mnt/extra-addons
      - ./etc:/etc/odoo
```

```bash
# Start
DB_PASSWORD=changeme docker compose up -d

# Access at http://localhost:8069
# First run: create database via web UI
```

### From source

```bash
# Clone
git clone https://github.com/hanzoai/suite.git
cd suite

# Install dependencies (use uv for venv)
uv venv
uv pip install -r requirements.txt

# Run
./odoo-bin --addons-path=addons -d mydb
```

## Core Concepts

### Architecture

```
                    Traefik (HTTPS)
                         |
              team.hanzo.ai / hanzo.team
                         |
                  +--------------+
                  |  Odoo 18.0   |  :8069 (web)
                  |  (Python)    |  :8072 (longpolling/livechat)
                  +--------------+
                         |
                  +--------------+
                  | PostgreSQL 16|
                  +--------------+

Odoo internals:
    odoo-bin (entrypoint)
        -> odoo.cli.main()
        -> odoo.http (Werkzeug-based HTTP server)
        -> odoo.models (ORM layer)
        -> odoo.fields (field definitions)
        -> odoo.api (environment, decorators)
        -> odoo/addons/* (built-in modules)
        -> /mnt/extra-addons/* (custom modules)
```

### Module System

Odoo's power comes from its modular architecture. Each module is a Python package with:

- `__manifest__.py` -- Module metadata (name, version, depends, data files)
- `models/` -- Python ORM model definitions
- `views/` -- XML view definitions (form, tree, kanban, etc.)
- `security/` -- Access control (ir.model.access.csv, record rules)
- `data/` -- Default data (XML/CSV)
- `static/` -- Web assets (JS, CSS, images)
- `controllers/` -- HTTP route handlers

### Odoo ORM

The ORM maps Python classes to PostgreSQL tables:

```python
from odoo import models, fields, api

class SaleOrder(models.Model):
    _name = 'sale.order'
    _description = 'Sales Order'

    name = fields.Char(string='Order Reference', required=True)
    partner_id = fields.Many2one('res.partner', string='Customer')
    order_line = fields.One2many('sale.order.line', 'order_id')
    amount_total = fields.Monetary(compute='_compute_total')
    state = fields.Selection([
        ('draft', 'Quotation'),
        ('sale', 'Sales Order'),
        ('done', 'Done'),
        ('cancel', 'Cancelled'),
    ], default='draft')

    @api.depends('order_line.price_subtotal')
    def _compute_total(self):
        for order in self:
            order.amount_total = sum(order.order_line.mapped('price_subtotal'))
```

### Key Built-in Modules

| Module | Purpose |
|--------|---------|
| `sale` | Sales orders and quotations |
| `purchase` | Purchase orders |
| `stock` | Warehouse and inventory management |
| `account` | Accounting and invoicing |
| `crm` | Customer relationship management |
| `project` | Project management and tasks |
| `hr` | Human resources |
| `website` | Website builder |
| `website_sale` | E-commerce storefront |
| `point_of_sale` | Point of sale |
| `mrp` | Manufacturing |
| `mail` | Internal messaging and email |

### Production Deployment

The production compose (`compose.prod.yaml`) uses:

- **Traefik** reverse proxy with automatic Let's Encrypt TLS
- **Domains**: `team.hanzo.ai` and `hanzo.team` (both routed to same Odoo instance)
- **CSP headers**: Scoped to `self` + `team.hanzo.ai` + `hanzo.team`
- **Named volumes**: `odoo-web-data-prod`, `odoo-db-data-prod`
- **Network**: `hanzo-network` (external Docker network shared with other Hanzo services)
- **Config**: Mounted from `/etc/hanzo/compose/odoo/files/config`
- **Custom addons**: Mounted from `/etc/hanzo/compose/odoo/files/addons`

## Directory structure

```
github.com/hanzoai/suite/
    odoo-bin                # Entrypoint script (Python)
    setup.py                # Package definition (pip installable as 'odoo')
    setup.cfg               # Setup configuration
    requirements.txt        # Pinned Python dependencies (per Python version)
    MANIFEST.in             # Package manifest
    compose.yaml            # Development Docker Compose
    compose.prod.yaml       # Production Docker Compose (Traefik + TLS)
    odoo/                   # Core framework
        __init__.py         # Framework initialization
        release.py          # Version info (18.0.0 final)
        api.py              # Environment, decorators (@api.depends, @api.model, etc.)
        fields.py           # Field type definitions (Char, Integer, Many2one, etc.)
        models.py           # Base Model, TransientModel, AbstractModel
        http.py             # HTTP layer (Werkzeug-based)
        sql_db.py           # Database connection pool
        exceptions.py       # UserError, ValidationError, AccessError, etc.
        netsvc.py           # Logging and network services
        addons/             # Built-in addon modules (100+)
        cli/                # Command-line interface (server, scaffold, shell, etc.)
        conf/               # Configuration handling
        modules/            # Module loader and registry
        service/            # RPC and WSGI services
        tests/              # Core framework tests
        tools/              # Utility functions (mail, image, misc, etc.)
        upgrade/            # Database upgrade scripts
        osv/                # Legacy ORM compatibility
    addons/                 # Custom/extra addons directory (mounted in Docker)
    debian/                 # Debian packaging files
    doc/                    # Documentation
    setup/                  # Setup scripts
    .github/                # Issue/PR templates
```

## Key Python Dependencies

| Package | Purpose |
|---------|---------|
| `psycopg2` | PostgreSQL adapter |
| `werkzeug` | HTTP/WSGI framework |
| `lxml` | XML/HTML processing |
| `Jinja2` | Template engine |
| `Pillow` | Image processing |
| `reportlab` | PDF generation |
| `passlib` | Password hashing |
| `gevent` | Async worker mode |
| `requests` | HTTP client |
| `cryptography` | Crypto operations |
| `python-ldap` | LDAP authentication (optional) |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "FATAL: role odoo does not exist" | PostgreSQL user not created | `createuser -s odoo` or set POSTGRES_USER in compose |
| Module not found | Addons path not set | Add `--addons-path=addons,/mnt/extra-addons` |
| "port 8069 already in use" | Another Odoo or service running | `lsof -i :8069` and stop conflicting process |
| Slow first startup | Module installation and asset compilation | Normal for first run; subsequent starts are faster |
| PDF reports blank | wkhtmltopdf missing | Install via package manager or use Docker image |
| "database does not exist" | No database created | Access `/web/database/manager` to create one |
| Assets not loading | CSS/JS not compiled | Clear browser cache; run `./odoo-bin --dev=assets` |

## Environment Variables

```bash
# Database connection
HOST=db              # PostgreSQL host
USER=odoo            # PostgreSQL user
PASSWORD=changeme    # PostgreSQL password

# Odoo config file (alternative to env vars)
# /etc/odoo/odoo.conf
```

## Related Skills

- `hanzo/hanzo-commerce.md` - Hanzo Commerce (web3 commerce, complementary to Suite)
- `hanzo/hanzo-platform.md` - PaaS for deploying Suite instances
- `hanzo/hanzo-database.md` - PostgreSQL infrastructure

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: erp, crm, odoo, accounting, inventory, e-commerce, python
**Prerequisites**: Python 3.10+, PostgreSQL 16, Docker (recommended)
