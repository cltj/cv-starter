# Roadmap: Workspace Activity Tracker

A laptop-only project that evolves from local object detection to a Databricks-connected data pipeline.

## Phase 1 — Detection Event Logging (local)

**Goal:** Turn real-time detections into structured, persistent data.

**What to build:**
- Extend `main.py` to log each detection as a structured event (timestamp, object class, confidence score, bounding box coordinates)
- Write events to a local file (CSV or JSON lines) — one row per detected object per frame
- Add frame sampling (e.g. log every 5 seconds, not every frame) to keep data volume sane
- Basic deduplication — don't log "laptop" 10 times if it hasn't moved

**Challenges:**
- Deciding what granularity to log at (every frame? every N seconds? only on change?)
- Object identity — YOLO detects "a cup" each frame, but is it the same cup? (intro to tracking)
- File format choices and their tradeoffs (CSV vs JSON vs Parquet)

**Core lessons:**
- Turning raw model output into structured data
- Event schema design — what fields matter and why
- Sampling strategies and their impact on data volume vs insight quality

---

## Phase 2 — Local Analysis & Querying

**Goal:** Extract insights from collected detection data without any cloud services.

**What to build:**
- Collect a few hours/days of workspace detection data
- Python scripts or notebooks to analyze: what objects appear most? when are you at your desk? what's your "desk signature"?
- Simple visualizations (matplotlib/plotly) — timeline charts, object frequency, presence heatmap

**Challenges:**
- Handling noisy data — false detections, flickering objects (detected one frame, gone the next)
- Smoothing and aggregation — raw detections are messy, useful insights require cleanup
- Defining meaningful metrics from raw detections

**Core lessons:**
- Data cleaning on ML output — models aren't perfect, your pipeline must handle that
- Aggregation patterns (raw events → per-minute summaries → hourly trends)
- Exploratory data analysis on self-generated data

---

## Phase 3 — Databricks Integration (batch)

**Goal:** Move detection data into Databricks and build a proper data pipeline.

**What to build:**
- Export detection logs as Parquet files
- Upload to Databricks (DBFS or cloud storage)
- Create a Delta Lake table from detection events
- Write Databricks SQL queries: daily desk usage, most common objects, activity patterns
- Build a simple Databricks dashboard

**Challenges:**
- Setting up Databricks workspace and storage connectivity
- Schema evolution — as you add fields to your detection events, Delta Lake must handle it
- Partitioning strategy — by date? by session? impacts query performance

**Core lessons:**
- Delta Lake fundamentals — ACID transactions, time travel, schema enforcement
- Databricks SQL for analytical queries on ML-generated data
- The value of a data platform: same data, but now queryable by anyone, versioned, governed
- Understanding the gap between "data on my laptop" and "data in a platform"

---

## Phase 4 — Streaming & Automation

**Goal:** Stream detection events to Databricks in near-real-time instead of batch uploads.

**What to build:**
- Detection events sent to a message queue or API endpoint as they happen
- Databricks Auto Loader or Structured Streaming to ingest events continuously
- Live-updating dashboard showing current workspace state
- Alerting — notify when specific conditions are met (e.g. "you've been away for 2 hours")

**Challenges:**
- Choosing a transport layer (REST API, message queue, cloud storage trigger)
- Handling connectivity issues — what happens when the network is down?
- Exactly-once vs at-least-once semantics in event delivery

**Core lessons:**
- Batch vs streaming ingestion — when to use which
- Auto Loader and Structured Streaming in Databricks
- Event-driven architecture patterns
- Reliability and idempotency in data pipelines

---

## Phase 5 — MLflow & Model Management

**Goal:** Track model experiments and manage the ML lifecycle through Databricks.

**What to build:**
- Log model runs to MLflow (which model, what settings, resulting FPS + detection quality)
- Compare experiments: nano vs small vs medium, different imgsz values, different confidence thresholds
- Register the "best" model configuration in MLflow Model Registry
- Track detection accuracy drift over time (does the model get worse in different lighting?)

**Challenges:**
- Defining "accuracy" without labeled ground truth (you don't have human-verified labels)
- Meaningful experiment design — what parameters actually matter?
- Model versioning when you're using pre-trained models (not training your own yet)

**Core lessons:**
- MLflow experiment tracking — the foundation of production ML
- Model registry and lifecycle stages (staging → production → archived)
- Monitoring model performance in the real world vs benchmarks
- Sets the stage for fine-tuning your own models later

---

## Phase 6 — Scale Simulation

**Goal:** Simulate what a multi-camera production deployment would look like on Databricks.

**What to build:**
- Record multiple video sessions from different angles/rooms
- Process them as if they were separate camera feeds
- Unified Delta Lake table with a `camera_id` column
- Cross-camera analytics: aggregate activity across "locations"
- Data quality monitoring — detect when a "camera" stops sending data

**Challenges:**
- Data modeling for multi-source ingestion
- Query performance at higher data volumes
- Building dashboards that work across multiple sources

**Core lessons:**
- Multi-tenant/multi-source data architecture
- The patterns that make a CV pipeline production-ready
- Data observability and monitoring
- Direct preview of what a real scaled deployment looks like

---

## Summary

| Phase | Focus | Key Tech | Output |
|-------|-------|----------|--------|
| 1 | Detection → data | Python, YOLO, file I/O | Local event log |
| 2 | Analysis | Pandas, matplotlib | Insights + visualizations |
| 3 | Data platform | Delta Lake, Databricks SQL | Dashboard on detection data |
| 4 | Streaming | Auto Loader, Structured Streaming | Live pipeline |
| 5 | ML lifecycle | MLflow, Model Registry | Tracked experiments |
| 6 | Scale | Multi-source ingestion | Production-ready architecture |

Each phase builds on the previous. You can pause after any phase and have something complete and useful.
