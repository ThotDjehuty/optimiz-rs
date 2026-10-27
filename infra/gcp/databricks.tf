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
  description         = "Bounded per-symbol optimizer risk snapshots; the optional analytics sync service imports completed manifests."
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
        input_table      = coalesce(var.databricks_input_table, "unset")
        output_volume    = coalesce(var.databricks_output_volume, "unset")
        max_rows         = "10000"
        periods_per_year = tostring(coalesce(var.databricks_periods_per_year, 0))
      }
    }
    library {
      whl = coalesce(var.optimizer_wheel_uri, "/Volumes/unset/unset/unset/unset.whl")
    }
  }
  lifecycle {
    precondition {
      condition     = var.optimizer_wheel_uri != null && var.databricks_input_table != null && var.databricks_output_volume != null && var.databricks_periods_per_year != null
      error_message = "Enabling the Databricks job requires the wheel, input table, output volume and explicit periods_per_year."
    }
  }
}
