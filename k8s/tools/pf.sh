#!/usr/bin/env bash
# Port forwarding access to postgres db
while true; do
kubectl port-forward svc/postgres 15432:5432
done
