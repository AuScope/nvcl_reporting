#!/usr/bin/env bash
# Script to start a manual database update
# Run this script from the "tools" directory
cd ../nvcl-report-app/templates
kubectl delete jobs manual-run-001
kubectl delete configmap my-script
kubectl create configmap my-script --from-file=script.sh
kubectl apply -f db-update-cronjob.yaml
kubectl create job --from=cronjob.batch/run-db-update manual-run-001
