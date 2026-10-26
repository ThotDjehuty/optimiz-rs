locals {
  create_job = var.databricks_cluster_id != null
}

resource "databricks_notebook" "export" {
  count    = local.create_job ? 1 : 0
  path     = "/Shared/${var.name}/export_analytics"
  language = "PYTHON"
  source   = "${path.module}/../../deploy/databricks/export_analytics.py"
}

resource "databricks_job" "export" {
  count               = local.create_job ? 1 : 0
  name                = "${var.name}-export"
  description         = "Manual bounded CSV export to an existing UC volume. Transfer completed files to GCE /data/datasets separately."
  max_concurrent_runs = 1
  timeout_seconds     = 1800
  task {
    task_key            = "export"
    existing_cluster_id = var.databricks_cluster_id
    timeout_seconds     = 1800
    max_retries         = 0
    notebook_task {
      notebook_path = databricks_notebook.export[0].path
      base_parameters = {
        input_table   = coalesce(var.databricks_input_table, "unset")
        output_volume = coalesce(var.databricks_output_volume, "unset")
        max_rows      = "10000"
      }
    }
    library {
      whl = coalesce(var.optimizer_wheel_uri, "/Volumes/unset/unset/unset/unset.whl")
    }
  }
  lifecycle {
    precondition {
      condition     = var.optimizer_wheel_uri != null && var.databricks_input_table != null && var.databricks_output_volume != null
      error_message = "Enabling the Databricks job requires optimizer_wheel_uri, databricks_input_table and databricks_output_volume."
    }
  }
}
