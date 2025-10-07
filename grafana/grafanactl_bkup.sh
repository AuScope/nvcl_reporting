#!/usr/bin/env bash
#
# Script to perform logical backup of Grafana dashboards and datasources
# This uses new grafanactl tool to export the config as JSON files
# NB: It does not export datasources
# From: https://grafana.github.io/grafanactl/
#
grafanactl config set contexts.dev.grafana.server http://localhost:3000
grafanactl config set contexts.dev.grafana.org-id 1
grafanactl config set contexts.dev.grafana.token <TOKEN GOES HERE>
grafanactl config use-context dev
grafanactl resources pull --path grafanactl_bkup
