resource "google_service_account" "analytics" {
  account_id   = var.name
  display_name = "Optimiz analytics VM"
}

resource "google_secret_manager_secret_iam_member" "token" {
  project   = var.project_id
  secret_id = var.analytics_token_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.analytics.email}"
}

resource "google_secret_manager_secret_iam_member" "sync" {
  count     = var.databricks_sync_secret_id != null ? 1 : 0
  project   = var.project_id
  secret_id = var.databricks_sync_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.analytics.email}"
}

resource "google_storage_bucket" "archive" {
  name                        = var.archive_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  versioning {
    enabled = true
  }
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_storage_bucket_iam_member" "archive_writer" {
  bucket = google_storage_bucket.archive.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.analytics.email}"
}

resource "google_compute_disk" "data" {
  name = "${var.name}-data"
  type = "pd-balanced"
  zone = var.zone
  size = var.data_disk_gb
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_compute_firewall" "iap_ssh" {
  name                    = "${var.name}-iap-ssh"
  network                 = var.network
  source_ranges           = ["35.235.240.0/20"]
  target_service_accounts = [google_service_account.analytics.email]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_instance" "analytics" {
  name                      = var.name
  zone                      = var.zone
  machine_type              = var.machine_type
  deletion_protection       = true
  allow_stopping_for_update = true
  can_ip_forward            = false
  boot_disk {
    initialize_params {
      image = var.cos_image
      size  = 20
    }
  }
  attached_disk {
    source      = google_compute_disk.data.id
    device_name = "analytics-data"
  }
  network_interface {
    subnetwork = var.subnetwork
    # Deliberately no access_config: this VM has no public IP.
  }
  service_account {
    email  = google_service_account.analytics.email
    scopes = ["cloud-platform"]
  }
  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }
  metadata = {
    enable-oslogin         = "TRUE"
    block-project-ssh-keys = "TRUE"
    serial-port-enable     = "FALSE"
    google-logging-enabled = "false"
    # Metadata updates preserve the VM/disk. Stop/start the VM after changing
    # image, token version or collector configuration to execute the new script.
    startup-script = templatefile("${path.module}/startup.sh.tftpl", {
      analytics_image     = var.analytics_image
      project_id          = var.project_id
      secret_id           = var.analytics_token_secret_id
      secret_version      = var.analytics_token_secret_version
      sync_enabled        = var.databricks_sync_secret_id != null
      sync_secret_id      = var.databricks_sync_secret_id == null ? "" : var.databricks_sync_secret_id
      sync_secret_version = var.databricks_sync_secret_version == null ? "" : var.databricks_sync_secret_version
      export_volume       = var.databricks_output_volume == null ? "" : var.databricks_output_volume
      collector           = file("${path.module}/../../deploy/otel/collector.yaml")
    })
  }
  lifecycle {
    precondition {
      condition = var.databricks_sync_secret_id == null || (
        var.databricks_sync_secret_version != null &&
        can(regex("^/Volumes/[A-Za-z0-9_]+/[A-Za-z0-9_]+/[A-Za-z0-9_]+$", var.databricks_output_volume))
      )
      error_message = "Sync requires a pinned credential secret version and an export volume with simple catalog/schema/volume identifiers."
    }
  }
  depends_on = [google_secret_manager_secret_iam_member.token, google_secret_manager_secret_iam_member.sync]
}
