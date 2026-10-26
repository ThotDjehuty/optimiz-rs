terraform {
  required_version = "= 1.13.5"
  backend "gcs" {}
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 7.0.1"
    }
    databricks = {
      source  = "databricks/databricks"
      version = "= 1.139.1"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Use Databricks unified authentication environment variables, never tfvars.
provider "databricks" {
  host = var.databricks_host
}
