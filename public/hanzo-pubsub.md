---
name: hanzo-pubsub
description: "High-performance event streaming and message queues with Hanzo PubSub (NATS). Pub/sub, persistent streams, consumer groups, exactly-once delivery."
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
              "package": "nats-py",
              "label": "Install NATS Client (pip)",
            },
          ],
      },
  }
---

# Hanzo PubSub — Event Streaming

`pip install nats-py`

NATS-based high-performance event streaming: pub/sub, persistent streams, consumer groups, exactly-once delivery. 10M+ messages/sec.

## Quick Start

```python
import nats

nc = await nats.connect("nats://localhost:4222")

# Simple pub/sub
await nc.publish("events.user.created", b'{"user_id": "123"}')

# Subscribe
async def handler(msg):
    print(f"Received: {msg.data.decode()}")

sub = await nc.subscribe("events.>", cb=handler)
```

## JetStream (Persistent Streams)

```python
js = nc.jetstream()

# Create persistent stream
await js.add_stream(name="ORDERS", subjects=["orders.*"])

# Publish to stream
ack = await js.publish("orders.new", b'{"item": "widget", "qty": 5}')
print(f"Sequence: {ack.seq}")

# Subscribe with durable consumer
sub = await js.subscribe("orders.*", durable="order-processor")
async for msg in sub.messages:
    print(f"Order: {msg.data.decode()}")
    await msg.ack()
```

## Consumer Groups

```python
# Create pull-based consumer group
psub = await js.pull_subscribe("orders.*", durable="workers")

# Fetch batch of messages
msgs = await psub.fetch(batch=10, timeout=5)
for msg in msgs:
    process(msg.data)
    await msg.ack()
```

## Key-Value Store (Built-in)

```python
kv = await js.create_key_value(bucket="config")

await kv.put("db.host", b"localhost")
entry = await kv.get("db.host")
print(entry.value.decode())

# Watch for changes
watcher = await kv.watchall()
async for entry in watcher:
    print(f"{entry.key} = {entry.value}")
```

## Object Store

```python
obs = await js.create_object_store(bucket="artifacts")

# Store object
await obs.put("model.bin", open("model.bin", "rb"))

# Retrieve
data = await obs.get("model.bin")
```

## Request/Reply

```python
# Service
async def handler(msg):
    await msg.respond(b"pong")

await nc.subscribe("ping", cb=handler)

# Client
response = await nc.request("ping", b"", timeout=5)
print(response.data.decode())  # "pong"
```

## Ports

- Client: `4222`
- Monitoring: `8222`

## Environment Variables

```bash
NATS_URL=nats://localhost:4222
NATS_TOKEN=...                  # Auth token
```
