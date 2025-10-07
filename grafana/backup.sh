#!/usr/bin/env bash
#
# Script to perform logical backup of Grafana dashboards and datasources
# This uses grizzly tool to export the config as JSON files
# From:  https://grafana.github.io/grizzly/workflows/
#
grr config create-context source
grr config set grafana.url http://localhost:3000
grr config set grafana.token <insert API token here>
grr config set targets Dashboard,Dashboardfolder,Datasource
grr pull resources
