# Grafana Service Docker Compose Notes

*NB:* Assumes docker and docker compose are installed


### Create docker volumes

The docker compose file requires two docker volumes to hold Grafana configuration and data

* Start docker container, the first time it starts the username is 'admin' password is 'admin'
```
docker run -d -p 3000:3000 --name=grafana grafana/grafana-enterprise
```

* Create docker volumes with user id 472 for grafana user
```
docker volume create grafana-config
docker cp grafana:/etc/grafana ./grafana-config-backup
docker run --rm -v grafana-config:/config -v "$(pwd)/grafana-config-backup:/config-backup" alpine   sh -c "cp -r /config-backup/* /config/ && chown -R 472 /config"

docker volume create grafana-data
docker cp grafana:/var/lib/grafana ./grafana-data-backup
docker run --rm -v grafana-data:/data -v "$(pwd)/grafana-data-backup:/data-backup" alpine   sh -c "cp -r /data-backup/* /data/ && chown -R 472 /data"
```

* Stop docker container
```
docker stop grafana
```

* Start docker container again
```
docker start grafana
```

* Test docker volumes, start Grafana, browse to 'https://hostname:3000', check logs for errors
```
docker run -d \
  --name=grafana-test \
  -p 3000:3000 \
  -v grafana-config:/etc/grafana \
  -v grafana-data:/var/lib/grafana \
  grafana/grafana-enterprise
```

* To run bash to edit or view files in volume
```
docker run --rm -it -v grafana-config:/config alpine /bin/ash
run --rm -it -v grafana-data:/data alpine /bin/ash
```

### Adding nvcl_reporting database to 'docker-compose.yml' file

* Edit the 'device:' line, add path to the directory where the nvcl db created by 'nvcl_reporting' resides
* It will be seen in '\nvcl-db' directory in Grafana UI and can be used by 'SQlite' plugin

### Start and Stop Grafana docker compose service

* Start grafana
```
cd docker
docker compose up
```

* Stop grafana
```
cd docker
docker compose down
```

### Backup Grafana docker volumes

* Script copies out volume contents into the local directory in the host filesystem as a .tar.gz file
```
cd docker
./backup.sh
```
