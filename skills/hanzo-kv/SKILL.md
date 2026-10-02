---
name: hanzo-kv
description: "Redis-compatible key-value store for caching, sessions, pub/sub messaging, streams, and real-time data. Sub-millisecond reads/writes."
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
              "package": "redis",
              "label": "Install Redis Client (pip)",
            },
          ],
      },
  }
---

# Hanzo KV — Key-Value Store

`pip install redis`

Redis-compatible high-performance in-memory data store. Sub-millisecond reads/writes for caching, sessions, pub/sub, and streaming.

## Quick Start

```python
import redis

client = redis.Redis(host="localhost", port=6379)

# Basic operations
client.set("key", "value")
value = client.get("key")

# With TTL
client.setex("session:abc", 3600, "user-data")

# Hash maps
client.hset("user:1", mapping={"name": "Alice", "email": "alice@hanzo.ai"})
user = client.hgetall("user:1")
```

## Async Client

```python
import redis.asyncio as aioredis

client = aioredis.Redis(host="localhost", port=6379)

await client.set("key", "value")
value = await client.get("key")
```

## Pub/Sub Messaging

```python
# Publisher
client.publish("events", '{"type": "user.created", "id": "123"}')

# Subscriber
pubsub = client.pubsub()
pubsub.subscribe("events")
for message in pubsub.listen():
    if message["type"] == "message":
        print(message["data"])
```

## Streams (Event Sourcing)

```python
# Add to stream
client.xadd("orders", {"item": "widget", "qty": "5"})

# Read from stream
messages = client.xread({"orders": "0"}, count=10)

# Consumer groups
client.xgroup_create("orders", "processors", "0")
messages = client.xreadgroup("processors", "worker-1", {"orders": ">"}, count=5)
client.xack("orders", "processors", message_id)
```

## Lists (Queues)

```python
# Push to queue
client.lpush("tasks", "task-1", "task-2")

# Pop from queue (blocking)
task = client.brpop("tasks", timeout=30)
```

## Sorted Sets (Leaderboards)

```python
client.zadd("scores", {"alice": 100, "bob": 85, "carol": 92})
top = client.zrevrange("scores", 0, 9, withscores=True)
```

## Lua Scripting

```python
script = client.register_script("""
    local current = redis.call('GET', KEYS[1])
    if current == false then
        redis.call('SET', KEYS[1], ARGV[1])
        return 1
    end
    return 0
""")
result = script(keys=["lock:resource"], args=["locked"])
```

## Port

- Default: `6379`

## Environment Variables

```bash
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=...              # Optional
```
