#!/usr/bin/env bash
#
# Script to perform logical backup of Grafana dashboards and datasources
# This uses deprecated grizzly tool to export the config as JSON files because new tool does not export datasources
# From:  https://grafana.github.io/grizzly/workflows/
#
grr config create-context source
grr config set grafana.url http://localhost:3000
grr config set grafana.token <TOKEN GOES HERE>
grr config set targets Dashboard,Dashboardfolder,Datasource
grr pull grizzly_bkup
