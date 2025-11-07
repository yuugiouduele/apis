# ETL Pipeline (Mermaid) — High level

```mermaid
flowchart TD
    subgraph Scheduler
        A[Scheduler (Cron / EventBridge / GitHub Actions)]
    end

    A --> B[Download Job Queue (Kubernetes Jobs / Batch)]
    B --> C{Download Nodes Pool}
    C --> D1[Downloader Pod (local SSD / tmpfs)]
    C --> D2[Downloader Pod (local SSD / tmpfs)]
    C --> D3[Downloader Pod (local SSD / tmpfs)]

    D1 --> E[Object Storage (GCS / S3) -- raw/]
    D2 --> E
    D3 --> E

    E --> F[Processing Node Pool (K8s) - ETL Workers]
    F --> G[Intermediate Storage (GCS / local SSD shards)]
    G --> H[BigQuery Load Jobs / Partitioned Tables]
    H --> I[2nd-order Stats / Aggregation (BigQuery)]
    I --> J[API Layer (API Gateway -> App Engines / Cloud Run)]
    J --> K[Client / Dashboard / Consumer]

    %% Monitoring & Scaling
    subgraph Observability
        M[Prometheus + Node Exporter] --> Grafana[Grafana + Alertmanager]
        Logs[(Structured Logs -> ELK / OpenSearch)]
    end
    D1 --> Logs
    F --> Logs
    E --> Logs
    H --> Logs
    M --> Grafana
    Grafana --> AutoScale[Cluster Autoscaler / Karpenter]

    %% Security/Network
    J -->|throttle / WAF| WAF[Cloud WAF / Load Balancer]
    WAF --> RateLimit[API Rate Limit / AuthN]
```