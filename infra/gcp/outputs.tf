output "instance_name" {
  value = google_compute_instance.analytics.name
}

output "private_ip" {
  value = google_compute_instance.analytics.network_interface[0].network_ip
}

output "preview_tunnel" {
  description = "Requires OS Login + IAP tunnel permissions and serviceAccountUser on the VM identity; use the app's bearer-token authentication."
  value       = "gcloud compute ssh ${var.name} --project=${var.project_id} --zone=${var.zone} --tunnel-through-iap -- -N -L 8080:127.0.0.1:8080"
}

output "archive_bucket" {
  description = "Private archive destination; this configuration does not synchronize UC volume exports or live files."
  value       = google_storage_bucket.archive.url
}

output "databricks_job_id" {
  value = try(databricks_job.export[0].id, null)
}
