---
name: hanzo-documentdb
description: "MongoDB-compatible document database with Hanzo DocumentDB (FerretDB). Store, query, and aggregate JSON documents with ACID guarantees."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [{ "id": "pip", "kind": "pip", "package": "pymongo", "label": "Install PyMongo (pip)" }],
      },
  }
---

# Hanzo DocumentDB — Document Database

`pip install pymongo`

MongoDB-compatible document database built on PostgreSQL via FerretDB. Full MongoDB wire protocol with ACID compliance.

## Quick Start

```python
from pymongo import MongoClient

client = MongoClient("mongodb://user:password@localhost:27017/")
db = client["myapp"]
collection = db["users"]

# Insert
result = collection.insert_one({"name": "Alice", "email": "alice@hanzo.ai", "role": "admin"})
print(f"Inserted: {result.inserted_id}")

# Find
user = collection.find_one({"email": "alice@hanzo.ai"})
print(user)

# Query
admins = collection.find({"role": "admin"}).sort("name", 1).limit(10)
for admin in admins:
    print(admin["name"])
```

## Async Client

```python
from motor.motor_asyncio import AsyncIOMotorClient

client = AsyncIOMotorClient("mongodb://user:password@localhost:27017/")
db = client["myapp"]

result = await db.users.insert_one({"name": "Bob"})
user = await db.users.find_one({"name": "Bob"})
```

## Aggregation Pipelines

```python
pipeline = [
    {"$match": {"status": "active"}},
    {"$group": {"_id": "$department", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
results = collection.aggregate(pipeline)
for doc in results:
    print(f"{doc['_id']}: {doc['count']}")
```

## Bulk Operations

```python
from pymongo import InsertOne, UpdateOne, DeleteOne

operations = [
    InsertOne({"name": "Carol"}),
    UpdateOne({"name": "Alice"}, {"$set": {"role": "superadmin"}}),
    DeleteOne({"name": "old-user"})
]
result = collection.bulk_write(operations)
```

## Indexes

```python
# Create index for fast queries
collection.create_index("email", unique=True)
collection.create_index([("name", 1), ("created_at", -1)])

# Text search index
collection.create_index([("title", "text"), ("body", "text")])
results = collection.find({"$text": {"$search": "hanzo agent"}})
```

## Transactions

```python
with client.start_session() as session:
    with session.start_transaction():
        db.accounts.update_one({"_id": "A"}, {"$inc": {"balance": -100}}, session=session)
        db.accounts.update_one({"_id": "B"}, {"$inc": {"balance": 100}}, session=session)
```

## Port

- Default: `27017` (MongoDB wire protocol)

## Environment Variables

```bash
MONGO_URI=mongodb://user:password@localhost:27017/database
```
