Vendor-pinned analytics deployment
==================================

This optional deployment layer does not change the optimizer's public API.
It is an analytics foundation, not an order-execution system. A production
deployment still requires vendor entitlements, GCP and Databricks accounts,
credentials, infrastructure review, and operational acceptance.

Where each component is used
----------------------------

* **GitHub** holds application code, vendor revision locks, infrastructure
  definitions, and review history. GitHub Actions validates and packages releases.
* **Docker** packages the analytics service and the compiled optimizer into a
  repeatable runtime. Development containers are not production web servers.
* **Docker Hub** stores the built application image. It does not run the service
  or store market data. Promote a tested image by its immutable digest.
* **Terraform** provisions cloud resources and Databricks job configuration.
  It runs before the application is deployed, and again when infrastructure
  changes. It is not a tick processor or a replacement for Docker.
* **SpiderRock** supplies licensed market data through its own SDK and interfaces.
  It is not exposed directly to a browser.
* **Databricks on GCP** queries stored datasets and runs historical analytics.
  Install the optimizer wheel on a compatible cluster; an arbitrary Docker Hub
  image is not automatically a supported Databricks runtime.
* **FINOS Perspective** renders interactive market-data and result grids.
* **OpenTelemetry** instruments the application and collects telemetry.
  **Prometheus** stores metrics; **Grafana** displays operational dashboards.
  Market data belongs in the data pipeline, not in telemetry labels.

Step 0: retain the complete vendor implementation
------------------------------------------------

The official source is `SpiderRock/SpiderRock-Connect-SDK
<https://github.com/SpiderRock/SpiderRock-Connect-SDK>`_.
The Git submodule and ``vendor-lock.json`` pin commit
``141b9388bb09121e49d3b7ae272c056817426203``. No sparse checkout excludes
examples. The vendor's nested submodules must also be initialized.

From the repository root::

    git submodule update --init --recursive
    python scripts/verify_vendor.py

The verifier rejects a changed source URL, floating revision, different gitlink,
missing interface family, dirty vendor working tree, or uninitialized/drifted
nested submodule. ``--metadata-only`` is available for inspecting a checkout
without vendor files, but is deliberately insufficient for release validation.

Do not edit the vendor directory, install dependencies into it, or generate
bytecode there. Keep adapters in this repository and use ``PYTHONDONTWRITEBYTECODE=1``
when importing vendor Python modules. Vendor updates require a reviewed change
to both the Git submodule revision and lock file followed by validation. The
check prevents accidental drift; authorized reviewers can intentionally change
the policy itself, so protect the deployment environment and require reviews.

The SDK contains trading and credential-placeholder examples. Retaining every
example does not mean running every example. No workflow should automatically
execute FIX or order-entry examples. Review each component's license and your
market-data agreement before redistributing vendor code or displaying its data.

Verified SpiderRock choices
--------------------------

The following families are listed by the pinned SDK's own root README:

.. list-table::
   :header-rows: 1
   :widths: 18 43 39

   * - Interface
     - Vendor capability
     - Fit in this deployment
   * - MLink
     - SQL-like REST and WebSocket API; bidirectional
     - First choice for entitled snapshot/streaming analytics; use read-only requests
   * - SRSE
     - SpiderRock Storage Engine, queried using SQL clients
     - Server-side queries and research; connectivity and access depend on entitlement
   * - FIX
     - FIX 4.4 messaging; vendor recommends it for ATS liquidity providers
     - Separate execution project, not enabled by this analytics deployment
   * - SpiderStream
     - Low-latency C#, C++, and Python message feeds; no direct trading functionality
     - Alternative ingestion service when throughput/network requirements justify it
   * - HistoricalData / Snowflake
     - SQL access to historical datasets, with vendor C# examples
     - Licensed backfills into storage/Databricks; not an assumed native GCP feed
   * - Env-Sandbox
     - Integration-test environment examples
     - Test authorized integrations before production

MLink also retains the vendor's JSON, framed JSON, protobuf, C++ and C# examples;
SpiderStream's languages and all other examples remain in the pinned SDK.
These are examples and interface options, not a promise that every interface
is enabled for your account.

Use MLink first when its entitlements and latency meet the requirement.
Use the vendor historical interface for backfills, or visualize your existing
stored data independently of SpiderRock. Do not invent an alternative wire
schema or silently replace missing vendor functionality.

Step 1: read-only SpiderRock bridge
----------------------------------

``scripts/spiderrock_ingest.py`` imports the unmodified vendor function
``MLink/_python/PROTO/clientRest.py::parse_custom_protocol_response`` and its
generated protobuf classes. The application implements bounded HTTPS transport
and normalization, **not** SpiderRock framing or a replacement SDK.

The first supported feed is ``LiveImpliedQuoteAdj`` through REST ``getmsgs``.
This is periodic current-state sampling, **not a lossless WebSocket/tick stream**.
The vendor's standalone WebSocket examples contain authentication placeholders
and are not reusable production client abstractions. They are retained, but
are not silently patched or presented as operational.

Each exported record preserves the complete option key as ``symbol``, the
vendor calculation timestamp, receipt time, vendor revision and environment.
``price`` is the option bid/ask midpoint, labeled ``price_kind=option_midpoint``;
it is not an underlying price or a trade. Invalid two-sided prices fail the
batch. Repeated identical observations are suppressed within a bounded
in-process cache; deduplication state does not survive restart.

The bridge has fixed HTTPS vendor endpoints, a fixed read command/message
type, a single validated ticker filter, timeouts, response-size bounds and
disabled redirects/proxy inheritance. It suppresses vendor diagnostic output
and rejects decoder errors and missing/unsuccessful QueryResult status.
Only a market-data-only vendor credential should be used. REST credentials
are query parameters in the vendor protocol: never enable HTTP URL tracing,
proxy logs or debugging that records those requests.
The standalone bridge uses a POSIX interval timer to interrupt stalled reads;
run it on Linux/macOS or use its Linux container on other hosts.

Create a runtime secret file outside the checkout and set
``SPIDERROCK_API_KEY_FILE`` to it. From the repository root::

    python -m pip install -r scripts/spiderrock-requirements.txt
    mkdir -p /path/to/data/live
    PYTHONDONTWRITEBYTECODE=1 python scripts/spiderrock_ingest.py \
      --environment sandbox --ticker AAPL-NMS-EQT \
      --output /path/to/data/live/market.jsonl --once

Remove ``--once`` for polling (default every five seconds); select ``live``
only after checking entitlements and sandbox results. The sandbox may not
provide every analytical message or ticker. No authenticated vendor test was
performed as part of this implementation.

Output rotates at 32 MiB into one ``.1`` file. This is bounded dashboard
handoff storage, not archival retention. Run only one writer for each file.
Use a separate archive process for licensed historical retention and recovery.
The vendor parser's permissive framing and version-specific server schemas
remain an integration acceptance item; pinning alone does not certify server
compatibility.

For hosts without Python/Git (including COS), build the vendor-free runtime::

    docker build -f scripts/Dockerfile.spiderrock -t optimizr-spiderrock:local .

Run it with the approved, recursively initialized checkout mounted read-only
at ``/workspace``, the secret file at ``/run/secrets/spiderrock_key``, and the
shared data directory at ``/data``. Set
``SPIDERROCK_API_KEY_FILE=/run/secrets/spiderrock_key`` and pass the same CLI
arguments with ``--output /data/live/market.jsonl``. UID/GID 10001 must be able
to read the checkout/secret and write the data directory. Mount the whole
checkout, including Git metadata and nested submodules, so integrity validation
can run. This image contains dependencies only, not redistributed vendor code.

Step 2: build and run the visualization stack
-------------------------------------------

Build ``services/analytics/Dockerfile`` using the repository root as its context.
It builds the full optimizer wheel and bundles the official FINOS Perspective
JavaScript packages locally rather than loading third-party CDN scripts.
Vendor code and credentials are not included in this application image.

``deploy/compose.yaml`` runs the analytics app, OpenTelemetry Collector,
Prometheus and Grafana. Supply ``ANALYTICS_IMAGE``, ``ANALYTICS_TOKEN_FILE``
and ``GRAFANA_PASSWORD_FILE`` before starting it. Generate strong random
secrets outside Git; the application token must contain at least 32 characters.
Compose bind-mounted secret files must be readable by the corresponding
non-root container user; do not make the whole secret directory public.

From the repository root::

    docker build -f services/analytics/Dockerfile -t optimizr-analytics:local .
    export ANALYTICS_IMAGE=optimizr-analytics:local
    export ANALYTICS_TOKEN_FILE=/absolute/private/path/analytics-token
    export GRAFANA_PASSWORD_FILE=/absolute/private/path/grafana-password
    docker compose -f deploy/compose.yaml up -d

Open ``http://localhost:8080`` for the authenticated Perspective dashboard,
``http://localhost:3000`` for Grafana, and ``http://localhost:9090`` for
Prometheus. These ports bind to loopback only. The application has a public
liveness endpoint at ``/healthz``; datasets and WebSocket updates require
authentication. Do not publish this local HTTP configuration directly to
the internet.

The application contract is:

* ``DATA_DIR=/data`` with administrator-managed ``datasets/*.csv`` files.
* ``LIVE_JSONL_PATH=/data/live/market.jsonl`` for the bridge output.
* ``ANALYTICS_TOKEN`` or ``ANALYTICS_TOKEN_FILE`` for access control.
* ``COOKIE_SECURE=true`` for HTTPS; local Compose/IAP HTTP tunnels explicitly
  disable Secure cookies because their browser endpoint is loopback HTTP.
* ``PUBLIC_ORIGIN=https://your-host`` behind an authenticated TLS proxy.
* ``OTEL_EXPORTER_OTLP_ENDPOINT`` to enable OpenTelemetry export.

Step 3: visualize stored data as well as live updates
----------------------------------------------------

Place CSV files in the application's ``/data/datasets`` directory (the
``analytics_data`` Compose volume, or the persistent data disk on GCP).
The required columns are ``event_time``, ``symbol`` and ``price``:

* Timestamps must be ISO-8601 with timezone.
* Prices must be finite and positive.
* Dataset filenames must be simple identifiers ending in ``.csv``.
* Use consistent instrument identifiers and sampling intervals.

The dashboard lists datasets and loads bounded selections into Perspective.
It also displays the bounded live JSONL window and calls the existing optimizer
risk function for a selected symbol. The full library is installed, but this
UI does **not** pretend to expose every optimizer algorithm through one generic
endpoint. Additional algorithms need explicit, validated domain inputs.
Annualization assumptions must match the observations; irregular sampled
option quotes are not automatically daily investment returns.

CSV files are capped at 16 MiB, with at most 10,000 rows returned per dataset.
Larger archives should be filtered/aggregated in Databricks before export.
This initial dashboard does not query arbitrary SQL, read arbitrary server
paths, support Parquet directly, or provide a historical replay scheduler.

The optional Terraform-managed Databricks job installs the supplied optimizer
wheel on an existing Unity Catalog-capable cluster and executes the shared
``optimizr.analytics.risk_report`` contract. Every eligible symbol receives its
own metrics; there is no median, ranking, or automatic winner. Fewer than five
prices, duplicate observation timestamps, and mixed price kinds produce explicit
symbol-level errors. Undefined numerical metrics remain null.

The job requires explicit ``periods_per_year`` and refuses to silently truncate
a selection larger than 10,000 rows. Use a bounded research table/view with
reviewed sampling assumptions. It writes an immutable snapshot followed by a
SHA-256 manifest into the configured UC volume. The optional sync service
downloads completed manifests, verifies size/hash and recomputes the results
before publishing a CSV/report pair. Restarting sync does not duplicate a
previously imported manifest. Changed imported files fail verification.

To use the existing SR4P research schema, run the export job with notebook
parameters ``input_format=sr4p_quotes``, the completed ``session_id``, and
optionally ``ticker``. Set ``input_table`` to the verified
``silver_live_implied_quote_adj`` table (the research source declares
``sr4p_dev.sososo_pipe_v1.silver_live_implied_quote_adj``). The adapter retains
all six option-key components, requires the vendor calculation timestamp with
a timezone, and computes the positive two-sided option midpoint from
``obid``/``oask``. It does not replace missing timestamps with receipt time,
blend collectors, infer daily sampling, or claim executable P&L.

The exporter is a separate manual Databricks job: run it after the selected
research session finishes. Creating it does not attach it to an existing
collector job or certify that your workspace contains the configured table.
The default ``input_format=normalized`` retains the generic price-table input.

For automatic local transfer, create a private Databricks configuration file
outside Git, readable by UID 10001, containing a ``[DEFAULT]`` profile with
``host``, ``auth_type = oauth-m2m``, ``client_id`` and ``client_secret``.
Use an identity with only the required workspace and UC read permissions.
Set ``DATABRICKS_CONFIG_FILE`` to that file and
``DATABRICKS_EXPORT_VOLUME=/Volumes/catalog/schema/exports``, then run::

    docker compose --env-file deploy/.env -f deploy/compose.yaml \
      --profile databricks up -d

The sync container polls every 60 seconds, sharing ``analytics_data`` with the
dashboard. Imported datasets show their saved per-symbol reports automatically.
Reload the page to discover new exports. The sync process exits visibly on
verification/authentication failures and is restarted by its supervisor.
Use one sync writer per data directory and a dedicated export volume; archive
completed exports before its 1,000-manifest limit. Only the newest 100 local
datasets are listed. Retention and archival deletion are operator-managed.

The image uses ``/syncenv`` to isolate the Databricks SDK's protobuf dependency
from the vendor bridge and application dependencies. Do not combine those
environments or change the official SDK to resolve dependency conflicts.

Step 4: publish to Docker Hub
-----------------------------

The ``Analytics image and infrastructure`` GitHub Actions workflow validates
the vendor pin, Terraform configuration and Compose configuration, then builds
the application image without publishing on pull requests.

Manual ``publish=true`` is restricted to the repository's default branch and
the ``dockerhub`` environment. Configure required reviewers and these settings:

* Variables: ``DOCKERHUB_USERNAME`` and ``DOCKERHUB_REPOSITORY`` (namespace/name).
* Secret: ``DOCKERHUB_TOKEN`` with only the required push permission.

The workflow reports the image digest. Use that digest for deployment rather
than ``latest`` or a mutable tag. The generic SpiderRock runtime can be built
and distributed separately; do not republish the complete vendor SDK without
confirming the licenses for every component.

Step 5: provision GCP and Databricks with Terraform
-------------------------------------------------

``infra/gcp`` targets an **existing** project, VPC/subnet and Databricks GCP
workspace. It creates a private analytics VM with a persistent disk, a
least-privilege runtime service account, an IAP-only SSH firewall rule and a
private archival bucket. It optionally provisions the Databricks notebook/job
against your existing cluster. It does not create a Databricks account/workspace.
Supply a wheel built for that cluster's architecture and Python ABI. The
analytics image's Linux wheel relies on system OpenBLAS; Databricks must have
compatible native libraries, or receive an appropriately packaged wheel.
This configuration does not install cluster-level native dependencies.

Prepare these prerequisites before planning:

* Required GCP APIs, an existing subnet with Private Google Access and NAT
  egress, and an explicitly selected COS image.
* An existing Secret Manager secret/version containing the dashboard token.
  Terraform grants runtime access without reading the secret value.
* A separate access-controlled GCS bucket for Terraform state.
* GitHub workload identity federation constrained to this repository, approved
  branch and environment; use distinct plan and deployment identities.
* Databricks OAuth identity/permissions, and (if enabling the job) a compatible
  cluster, immutable optimizer wheel path, input table, output UC volume and
  explicit ``databricks_periods_per_year``.

For automatic GCE transfer, additionally set ``databricks_sync_secret_id`` and
``databricks_sync_secret_version`` to an existing Secret Manager secret/version
containing the read-only Databricks profile described above. The optional sync
systemd service uses the same analytics image and persistent data disk. Terraform
grants access only to that selected secret; its value is fetched at VM startup,
never placed in Terraform state or metadata. The sync identity is separate from
the job/deployment identity. Without these inputs, cloud sync remains disabled.

Build and install the wheel from this integration, not an older installed wheel
with the same package version: it must contain ``optimizr.snapshot``. Publish it
under an immutable release directory in the UC volume and verify the cluster
imports that module before running the job.

Use ``terraform.tfvars.example`` and ``backend.hcl.example`` as configuration
references. Supply real values outside committed files. Remote-state bucket
creation and authentication bootstrap are deliberately separate.

The manual workflow accepts ``terraform_operation=plan`` or ``apply``.
Configure protected ``analytics-plan`` and ``analytics-production`` environments.
Apply additionally requires ``confirm_apply=APPLY``. Configure:

* ``GCP_WORKLOAD_IDENTITY_PROVIDER``, ``GCP_TERRAFORM_SERVICE_ACCOUNT``.
* ``TF_STATE_BUCKET``, ``TF_STATE_PREFIX``.
* ``ANALYTICS_TFVARS_JSON`` containing non-secret Terraform inputs.
* ``DATABRICKS_CLIENT_ID`` and secret ``DATABRICKS_CLIENT_SECRET``.

Publishing and applying are separate promotions: first publish, then select the
tested digest in Terraform inputs, review the plan and authorize apply.
The starter VM can pull a public Docker Hub image; private registry credential
provisioning is not implemented.
After changing the configured image digest, token version or collector settings,
perform a controlled VM stop/start to execute the updated startup script.
Terraform metadata updates alone do not restart the running application.

The VM exposes the dashboard only on its loopback interface. Use IAP SSH
port forwarding to access it, for example::

    gcloud compute ssh YOUR_VM --project YOUR_PROJECT --zone YOUR_ZONE \
      --tunnel-through-iap -- -N -L 8080:127.0.0.1:8080

Then open ``http://localhost:8080`` locally. A public/shared production website
needs an additional reviewed HTTPS identity-aware ingress with compatible
WebSocket forwarding and secure cookies; no public ingress is created here.

Step 6: monitor, accept and operate
----------------------------------

The local Compose stack provisions a Grafana dashboard backed by Prometheus.
The VM starts the application and OpenTelemetry Collector; connect a secured
Prometheus/Grafana deployment to the collector's metrics endpoint using private
connectivity or a tunnel. The local monitoring stack is not automatically
installed as a managed cloud Grafana service.

This starter stores metrics, not distributed traces/logs: configuring an
OpenTelemetry exporter does not create a Tempo/Loki backend. Add official
backends and retention policies separately if those signals are required.
Enable ``OTEL_TRACES_EXPORTER=otlp`` only when the configured endpoint accepts
traces; the provided collector is metrics-only.

Before calling the deployment production-ready:

* Verify vendor credentials, entitlements, sandbox/live provenance and schema
  compatibility using real authorized responses.
* Verify live snapshots, stored CSV selection, optimizer results and Grafana
  metrics end to end.
* Exercise reconnects, file rotation, stale-feed visibility, malformed data,
  restart behavior and disk limits.
* Verify the configured Databricks sync identity and establish archival transfer, backups,
  alerts, identity-based access and incident ownership.
* Test rollbacks to the previous image digest and inspect Terraform plans;
  disk/bucket protections intentionally prevent accidental data destruction.

Nothing in this workflow enables trading, downloads your private datasets,
creates cloud resources during pull-request validation, or claims that
unconfigured infrastructure is already live.
