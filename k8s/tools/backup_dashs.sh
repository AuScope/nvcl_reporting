#!/usr/bin/env bash
# Script to backup grafana dashboards
# Requires "jq", e.g. for Ubuntu use "sudo apt install -y jq" to install

GRAFANA_URL="<grafana URL goes here>"
TOKEN="<grafana API token goes here>"
OUTDIR="./grafana_dashboards_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

curl -k -sS -H "Authorization: Bearer $TOKEN" \
  "$GRAFANA_URL/api/search?type=dash-db&limit=5000" \
| jq -r '.[] | [.uid, .title] | @tsv' \
| while IFS=$'\t' read -r uid title; do
    safe_title=$(echo "$title" | tr '/:' '__' | tr -cd '[:alnum:] _.-')
    curl -k -sS -H "Authorization: Bearer $TOKEN" \
      "$GRAFANA_URL/api/dashboards/uid/$uid" \
      > "$OUTDIR/${safe_title}__${uid}.json"
  done
