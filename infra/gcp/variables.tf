variable "project_id" {
  type        = string
  description = "Existing GCP project with Compute, Secret Manager, IAM and Storage APIs enabled."
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "Provide a valid 6-30 character GCP project ID."
  }
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  type    = string
  default = "us-central1-a"
  validation {
    condition     = startswith(var.zone, "${var.region}-")
    error_message = "zone must belong to region."
  }
}

variable "name" {
  type    = string
  default = "optimiz-analytics"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,23}[a-z0-9]$", var.name))
    error_message = "Use 6-25 lowercase letters, digits and hyphens, beginning with a letter and ending with a letter or digit."
  }
}

variable "subnetwork" {
  type        = string
  description = "Existing subnet self-link in this project's region; requires Private Google Access and NAT egress to Docker Hub."
}

variable "network" {
  type        = string
  description = "Existing VPC self-link used for the narrowly scoped IAP SSH rule."
}

variable "analytics_image" {
  type        = string
  description = "Publicly pullable Docker Hub analytics image pinned by digest; private registry login is not configured."
  validation {
    condition     = can(regex("^docker\\.io/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$", var.analytics_image))
    error_message = "Provide docker.io/namespace/repository@sha256:<64 lowercase hex characters>."
  }
}

variable "cos_image" {
  type        = string
  description = "Explicit COS LTS image self-link/name (not a mutable image family); verify availability in cos-cloud before deployment."
  validation {
    condition     = can(regex("^projects/cos-cloud/global/images/cos-[0-9][a-z0-9-]+$", var.cos_image))
    error_message = "Supply an explicit projects/cos-cloud/global/images/cos-... image."
  }
}

variable "machine_type" {
  type    = string
  default = "e2-standard-2"
}

variable "data_disk_gb" {
  type    = number
  default = 100
  validation {
    condition     = var.data_disk_gb >= 20 && floor(var.data_disk_gb) == var.data_disk_gb
    error_message = "Persistent disk size must be an integer of at least 20 GiB."
  }
}

variable "analytics_token_secret_id" {
  type        = string
  description = "Existing Secret Manager secret ID in project_id, containing the application's bearer token. Terraform never reads its value."
  validation {
    condition     = can(regex("^[A-Za-z0-9_-]+$", var.analytics_token_secret_id))
    error_message = "Provide only the secret ID, not a version or resource path."
  }
}

variable "analytics_token_secret_version" {
  type        = string
  description = "Existing numeric secret version; change this and restart the VM for deliberate token rotation."
  validation {
    condition     = can(regex("^[1-9][0-9]*$", var.analytics_token_secret_version))
    error_message = "Pin a numeric secret version."
  }
}

variable "archive_bucket_name" {
  type        = string
  description = "Globally unique name for the private GCS archive. Transport/backup scheduling is operator-managed, not automatic."
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9_-]{1,61}[a-z0-9]$", var.archive_bucket_name))
    error_message = "Use a globally unique 3-63 character bucket name with letters, digits, underscores or hyphens."
  }
}

variable "databricks_host" {
  type        = string
  description = "Existing GCP workspace HTTPS URL; configure DATABRICKS_AUTH_TYPE and OAuth credentials outside Terraform."
  validation {
    condition     = can(regex("^https://[a-zA-Z0-9.-]+\\.gcp\\.databricks\\.com/?$", var.databricks_host))
    error_message = "Expected an HTTPS GCP Databricks workspace URL."
  }
}

variable "databricks_cluster_id" {
  type        = string
  default     = null
  nullable    = true
  description = "Existing UC-capable cluster ID, or null to skip notebook/job creation. Runtime must support /Volumes and the wheel ABI."
}

variable "optimizer_wheel_uri" {
  type        = string
  default     = null
  description = "Immutable wheel path in an existing UC volume (/Volumes/catalog/schema/volume/file.whl), readable by the job identity."
  validation {
    condition     = var.optimizer_wheel_uri == null ? true : can(regex("^/Volumes/[^/]+/[^/]+/[^/]+/.+\\.whl$", var.optimizer_wheel_uri))
    error_message = "Use an existing Unity Catalog volume wheel path ending in .whl."
  }
}

variable "databricks_input_table" {
  type        = string
  default     = null
  description = "Existing catalog.schema.table with event_time (UTC timestamp), symbol and positive price columns; job identity requires SELECT."
  validation {
    condition     = var.databricks_input_table == null ? true : can(regex("^[A-Za-z_][A-Za-z0-9_]*\\.[A-Za-z_][A-Za-z0-9_]*\\.[A-Za-z_][A-Za-z0-9_]*$", var.databricks_input_table))
    error_message = "Use an unquoted catalog.schema.table identifier."
  }
}

variable "databricks_output_volume" {
  type        = string
  default     = null
  description = "Existing /Volumes/catalog/schema/volume; job identity requires WRITE VOLUME. Not mounted on the GCE VM."
  validation {
    condition     = var.databricks_output_volume == null ? true : can(regex("^/Volumes/[^/]+/[^/]+/[^/]+$", var.databricks_output_volume))
    error_message = "Use /Volumes/catalog/schema/volume without a trailing slash."
  }

}

variable "databricks_periods_per_year" {
  type        = number
  default     = null
  description = "Explicit annualization frequency of the normalized observations; no inferred daily sampling."
  validation {
    condition = var.databricks_periods_per_year == null ? true : (
      var.databricks_periods_per_year >= 1 && var.databricks_periods_per_year <= 1000000
    )
    error_message = "periods_per_year must be between 1 and 1000000."
  }
}

variable "databricks_sync_secret_id" {
  type        = string
  default     = null
  description = "Opt-in: existing Secret Manager secret containing a DEFAULT Databricks OAuth profile with read-only access to the export volume."
  validation {
    condition     = var.databricks_sync_secret_id == null ? true : can(regex("^[A-Za-z0-9_-]+$", var.databricks_sync_secret_id))
    error_message = "Provide a Secret Manager secret ID."
  }
}

variable "databricks_sync_secret_version" {
  type        = string
  default     = null
  description = "Pinned numeric version of the sync OAuth profile secret."
  validation {
    condition     = var.databricks_sync_secret_version == null ? true : can(regex("^[1-9][0-9]*$", var.databricks_sync_secret_version))
    error_message = "Pin a numeric secret version."
  }
}
