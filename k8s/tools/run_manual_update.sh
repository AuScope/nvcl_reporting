#!/usr/bin/env bash
# Script to start a manual database update
# 
# Edit "db-update-cronjob.sh" replacing variable references with working values
# Run this script from the "tools" directory
# 
cd ../nvcl-report-app/templates
kubectl delete jobs manual-run-001
kubectl delete configmap my-script
kubectl create configmap my-script --from-file=db_update.sh
kubectl apply -f db-update-cronjob.yaml
kubectl create job --from=cronjob.batch/run-db-update manual-run-001
