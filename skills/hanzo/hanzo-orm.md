# Hanzo ORM - Go Generics ORM with Auto-Serialization

**Category**: Hanzo Ecosystem
**Related Skills**: `hanzo/hanzo-database.md`, `hanzo/go-sdk.md`

## Overview

Hanzo ORM is a **Go generics-based ORM** with automatic serialization, KV caching, and zero code generation. 130 tests passing, supports SQLite (embedded) and ZAP (PostgreSQL/MongoDB/Redis/ClickHouse via sidecar).

### Why Hanzo ORM?

- **Zero codegen**: Generics-based, no code generation step
- **Auto-serialization**: Struct ↔ DB mapping automatic
- **KV cache layer**: Built-in caching for hot paths
- **Multi-backend**: SQLite embedded + ZAP for production DBs
- **Type-safe**: Full Go generics type safety

## When to use

- Go services needing database access
- Projects wanting ORM without code generation
- Embedded SQLite for simple apps
- Multi-backend with caching

## Quick reference

| Item | Value |
|------|-------|
| Module | `github.com/hanzoai/orm` |
| Version | v0.3.0 |
| Go | 1.26.0 |
| Tests | 130 passing (orm: 97, db: 21, val: 11) |
| Docs | `orm.hanzo.ai` → `hanzo.ai/docs/services/orm` |
| Repo | `github.com/hanzoai/orm` |

## One-file quickstart

```go
package main

import (
    "github.com/hanzoai/orm"
    "github.com/hanzoai/orm/db/sqlite"
)

type User struct {
    ID    int64  `orm:"pk,autoincr"`
    Name  string `orm:"notnull"`
    Email string `orm:"unique,notnull"`
    Age   int    `orm:"default:0"`
}

func main() {
    db, _ := sqlite.Open("app.db")
    repo := orm.NewRepository[User](db)

    // Create
    user := User{Name: "Alice", Email: "alice@hanzo.ai", Age: 30}
    repo.Create(&user)

    // Query
    found, _ := repo.FindByID(user.ID)
    all, _ := repo.FindAll(orm.Where("age > ?", 25))

    // Update
    found.Age = 31
    repo.Update(found)

    // Delete
    repo.Delete(found.ID)
}
```

### ZAP Backend (PostgreSQL)

```go
import "github.com/hanzoai/orm/db/zap"

db, _ := zap.Open("postgresql://user:pass@localhost:5432/mydb")
repo := orm.NewRepository[User](db)
// Same API, production database
```

## Backends

| Backend | Use Case | Module |
|---------|----------|--------|
| SQLite | Embedded, testing, small apps | `orm/db/sqlite` |
| ZAP-PostgreSQL | Production RDBMS | `orm/db/zap` |
| ZAP-MongoDB | Document store | `orm/db/zap` |
| ZAP-Redis | KV cache layer | `orm/db/zap` |
| ZAP-ClickHouse | Analytics/OLAP | `orm/db/zap` |

## Related Skills

- `hanzo/hanzo-database.md` - Database configuration
- `hanzo/go-sdk.md` - Go SDK patterns

---

**Last Updated**: 2026-03-13
**Category**: Hanzo Ecosystem
**Related**: orm, database, go, generics
**Prerequisites**: Go generics, SQL basics
