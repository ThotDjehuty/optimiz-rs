mock_provider "google" {}
mock_provider "databricks" {}

variables {
  project_id                     = "test-project"
  network                        = "projects/test-project/global/networks/private"
  subnetwork                     = "projects/test-project/regions/us-central1/subnetworks/private"
  cos_image                      = "projects/cos-cloud/global/images/cos-121-test"
  analytics_image                = "docker.io/example/analytics@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  analytics_token_secret_id      = "analytics-token"
  analytics_token_secret_version = "1"
  archive_bucket_name            = "test-private-archive"
  databricks_host                = "https://test.gcp.databricks.com"
}

run "private_vm_without_job" {
  command = plan
  assert {
    condition     = length(google_compute_instance.analytics.network_interface[0].access_config) == 0
    error_message = "The analytics VM must not have a public IP."
  }
  assert {
    condition     = google_compute_instance.analytics.deletion_protection
    error_message = "The VM must be protected against accidental deletion."
  }
  assert {
    condition     = google_storage_bucket.archive.public_access_prevention == "enforced"
    error_message = "Archives must not be public."
  }
  assert {
    condition     = length(databricks_job.export) == 0
    error_message = "Databricks job creation must be opt-in."
  }
  assert {
    condition     = strcontains(google_compute_instance.analytics.metadata["startup-script"], "-p 127.0.0.1:8080:8080")
    error_message = "The app must listen only on the VM loopback interface."
  }
}

run "existing_cluster_job" {
  command = plan
  variables {
    databricks_cluster_id       = "0000-000000-existing"
    optimizer_wheel_uri         = "/Volumes/catalog/schema/packages/optimizer.whl"
    databricks_input_table      = "catalog.schema.quotes"
    databricks_output_volume    = "/Volumes/catalog/schema/exports"
    databricks_periods_per_year = 252
  }
  assert {
    condition     = databricks_job.export[0].task[0].existing_cluster_id == "0000-000000-existing"
    error_message = "The export job must use the operator's existing cluster."
  }
  assert {
    condition     = databricks_job.export[0].task[0].notebook_task[0].base_parameters["output_volume"] == "/Volumes/catalog/schema/exports"
    error_message = "The notebook must receive the configured existing UC volume."
  }
}

run "incomplete_job_rejected" {
  command = plan
  variables {
    databricks_cluster_id = "0000-000000-existing"
  }
  expect_failures = [databricks_job.export]
}

run "mutable_image_rejected" {
  command = plan
  variables {
    analytics_image = "docker.io/example/analytics:latest"
  }

  expect_failures = [var.analytics_image]
}

run "sync_is_explicit_and_secret_scoped" {
  command = plan
  variables {
    databricks_sync_secret_id      = "databricks-reader-profile"
    databricks_sync_secret_version = "2"
    databricks_output_volume       = "/Volumes/catalog/schema/exports"
  }
  assert {
    condition     = google_secret_manager_secret_iam_member.sync[0].secret_id == "databricks-reader-profile"
    error_message = "Sync must have access only to the explicitly selected credential secret."
  }
  assert {
    condition     = strcontains(google_compute_instance.analytics.metadata["startup-script"], "--export-volume /Volumes/catalog/schema/exports --interval-seconds 60")
    error_message = "Sync must use the approved export volume."
  }
}

run "incomplete_sync_rejected" {
  command = plan
  variables {
    databricks_sync_secret_id = "databricks-reader-profile"
  }
  expect_failures = [google_compute_instance.analytics]
}
