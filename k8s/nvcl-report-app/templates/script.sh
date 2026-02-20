#!/usr/bin/env bash
MYUID=$1
echo "MYUID=$MYUID"
#
apt update -y
# Install git & curl
apt install -y git curl
#
# Install pyproj prerequisites
apt install -y python3 python3-pip gcc g++ build-essential libproj-dev proj-data proj-bin libgeos-dev

# Create new user
groupadd -g $MYUID batch
useradd -u $MYUID -m -s /bin/bash -g batch batch
cd /home/batch

## Create script START
cat << EOF2 > batch_install.sh
#!/usr/bin/env bash

#
# Clone repo
git clone --depth 1 https://github.com/AuScope/nvcl_reporting.git
cd nvcl_reporting
#
# Install pdm
curl -sSL https://pdm-project.org/install.sh | bash
export PATH="/root/.local/bin:$PATH"
#
# Install python packages
pdm install
#
# Create config file
cat << EOF > config.yaml
plot_dir: /nvcl-db/plots
db: /nvcl-db/DBs/nvcl16-kms.db
tsg_meta_file: /nvcl-db/DBs/metadata.csv
tmp_dir: /tmp
EOF
# Run db updates
cd src
pdm run make_reports.py -utb
EOF2
## Create script END

chmod +x batch_install.sh
chown batch: batch_install.sh

## Execute script as 'batch' user to allow NFS access
su batch -c ./batch_install.sh
