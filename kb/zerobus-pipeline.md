# Zerobus Pipeline — Producer to Databricks

## Architecture

```
Detector (producer)
    ↓ gRPC or REST
Zerobus Ingest API (managed by Databricks, serverless)
    ↓
Delta table: raw_cv.detections (Unity Catalog)
    ↓ (Databricks job or DLT pipeline)
Delta table: form_cv.detections (cleaned/enriched)
    ↓ (validation logic)
Delta table: valid_cv.detections (business-ready)
```

No Kafka, no message broker, no infrastructure to manage.

## What you need to set up

### 1. Databricks side (one-time setup)

**a) Get the Zerobus endpoint**

Format: `<workspace-id>.zerobus.<region>.cloud.databricks.com`

Found in your Databricks workspace settings.

**b) Create the target Delta table**

This is the raw landing table — first layer in your naming convention:

```sql
CREATE TABLE raw_cv.detections (
    timestamp       STRING,
    object_class    STRING,
    confidence      DOUBLE,
    bbox_x1         DOUBLE,
    bbox_y1         DOUBLE,
    bbox_x2         DOUBLE,
    bbox_y2         DOUBLE,
    session_id      STRING,
    device_id       STRING
);
```

For metrics:

```sql
CREATE TABLE raw_cv.metrics (
    timestamp       STRING,
    motion_pct      DOUBLE,
    yolo_ran        INT,
    inference_ms    DOUBLE,
    fps             DOUBLE,
    objects_detected INT,
    cpu_pct         DOUBLE,
    memory_mb       DOUBLE,
    yolo_total_sec  DOUBLE,
    session_id      STRING,
    device_id       STRING
);
```

**c) Create a service principal (access control)**

This is how the detector authenticates — it doesn't use your personal credentials.

1. Go to Settings → Identity and Access → Service principals
2. Click "Add service principal"
3. Generate an OAuth secret (save the client ID + secret)

**d) Grant permissions (least privilege)**

```sql
GRANT USE CATALOG ON CATALOG raw_cv TO `<service-principal-uuid>`;
GRANT USE SCHEMA ON SCHEMA raw_cv.detections TO `<service-principal-uuid>`;
GRANT MODIFY, SELECT ON TABLE raw_cv.detections TO `<service-principal-uuid>`;
GRANT MODIFY, SELECT ON TABLE raw_cv.metrics TO `<service-principal-uuid>`;
```

The service principal can ONLY write to these specific tables. Nothing else.

### 2. Producer side (our detector)

**a) Install the SDK**

```bash
uv add databricks-zerobus-ingest-sdk
```

**b) Configure credentials (never hardcode)**

Store in environment variables or a `.env` file (already in .gitignore):

```bash
# .env
ZEROBUS_ENDPOINT=<workspace-id>.zerobus.<region>.cloud.databricks.com
DATABRICKS_WORKSPACE_URL=https://<instance>.cloud.databricks.com
DATABRICKS_CLIENT_ID=<service-principal-uuid>
DATABRICKS_CLIENT_SECRET=<oauth-secret>
```

**c) Send detections**

```python
from zerobus.sdk.sync import ZerobusSdk
from zerobus.sdk.shared import RecordType, StreamConfigurationOptions, TableProperties

sdk = ZerobusSdk(ZEROBUS_ENDPOINT, DATABRICKS_WORKSPACE_URL)
table_props = TableProperties("raw_cv.detections")
options = StreamConfigurationOptions(record_type=RecordType.JSON)
stream = sdk.create_stream(CLIENT_ID, CLIENT_SECRET, table_props, options)

# In the detection loop:
record = {
    "timestamp": "2026-07-21T10:30:00+00:00",
    "object_class": "person",
    "confidence": 0.89,
    "bbox_x1": 100.0, "bbox_y1": 200.0,
    "bbox_x2": 300.0, "bbox_y2": 400.0,
    "session_id": "20260721_103000",
    "device_id": "laptop-01",
}
stream.ingest_record(record)

# On shutdown:
stream.close()
```

### 3. Downstream layers (Databricks jobs or DLT)

**raw → form (cleaning)**

```sql
-- form_cv.detections
CREATE OR REFRESH MATERIALIZED VIEW form_cv.detections AS
SELECT
    CAST(timestamp AS TIMESTAMP) AS timestamp,
    object_class,
    ROUND(confidence, 3) AS confidence,
    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
    session_id,
    device_id,
    current_timestamp() AS processed_at
FROM raw_cv.detections
WHERE confidence >= 0.5
  AND object_class IS NOT NULL;
```

**form → valid (business rules)**

```sql
-- valid_cv.detections
CREATE OR REFRESH MATERIALIZED VIEW valid_cv.detections AS
SELECT *
FROM form_cv.detections
WHERE object_class IN ('person', 'laptop', 'keyboard', 'mouse',
                        'cell phone', 'cup', 'bottle', 'book');
```

## gRPC vs REST — which to use

| | gRPC (SDK) | REST API |
|--|-----------|----------|
| Use when | Continuous stream (our detector) | Occasional events, webhooks |
| Throughput | High (persistent connection) | Lower (handshake per request) |
| Latency | Sub-5 seconds | Slightly higher |
| Code | SDK handles batching, auth refresh | Manual HTTP calls |
| Endpoint | Via SDK | `POST /zerobus/v1/tables/<table>/insert` |

**For our project: use gRPC SDK.** The detector runs continuously, so a persistent connection makes sense.

## Security checklist

- [ ] Service principal with least-privilege access (only MODIFY on target tables)
- [ ] OAuth secret stored in environment variables, never in code
- [ ] .env file in .gitignore
- [ ] Service principal cannot read other tables or create resources
- [ ] Rotate OAuth secret periodically
