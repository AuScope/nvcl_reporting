# Kubernetes README

## "nvcl_report_app" directory

Contains helm chart files as follows.

NB: The K8s namespace in many files is "nvcl-projects". This can be altered as required.


#### 1. Database Update

**db-update-cronjob.yaml** - CronJob to update postgres database. It has three mounts:

  1. "scripts" - created at run time to access "script.sh"
  2. "nvcl-fs" - output from update save to NFS mount e.g. CSV files derived from TSG files, plot files
  3. "nvcl-disk" - storage for update

**script.sh** - a script which creates another script which runs on pod and updates the database

**nvcl-pvc.yaml** - storage for update

#### 2. Grafana

**grafana-deployment.yaml** - Grafana Enterprise in a docker container. It also has three mounts:

  1. "grafana-config"  - grafana config
  2. "grafana-data" - grafana data
  3. "grafana-datasources" - a mounted secret containing postgres connection parameters (See Section 4)

Grafana data and configuration are stored on NFS mounts

**grafana-svc.yaml** - service for grafana

**ingress.yaml** - ingress for grafana web service


#### 3. Postgres Database

**pg-db.yaml** - postgres database in a docker container, has one mount for the PVC

**pg-pvc.yaml** - storage for postgres data

**pg-svc.yaml** - service for postgres

#### 4. Secrets files

Example secrets file for the database connections. 
Replace "\<db-name-here\>", "\<db-user-here\>", "\<db-password-here\>" with suitable values

**grafana-pg-secret.yaml** - used by grafana to connect to postgres

**pg-secret.yaml** - used by postgres db


## "tools" directory

* Contains scripts for

  - Running a manual database update
  - Backup of dashboard files
  - Set up port forwarding, for direct access to database 
