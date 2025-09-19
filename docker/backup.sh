#!/bin/bash -x

TODAY=`date +%Y%m%d`

docker run --rm \
  -v grafana-config:/data \
  -v $(pwd):/backup \
  alpine \
  tar czf /backup/$TODAY-grafana-config-vol-bkup.tar.gz -C /data .

docker run --rm \
  -v grafana-data:/data \
  -v $(pwd):/backup \
  alpine \
  tar czf /backup/$TODAY-grafana-data-vol-bkup.tar.gz -C /data .

