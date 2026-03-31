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

echo "Creating 'batch' user"

# Create new user called "batch" with UID that can access NFS
groupadd -g $MYUID batch
useradd -u $MYUID -m -s /bin/bash -g batch batch
cd /home/batch

echo "Creating batch script"

## Create script START
cat << EOF2 > batch_install.sh
#!/usr/bin/env bash

#
# Clone repo
git clone -b postgres-migrate --depth 1 https://github.com/AuScope/nvcl_reporting.git
cd nvcl_reporting
#
# Install pdm
curl -sSL https://pdm-project.org/install.sh | bash
export PATH="/home/batch/.local/bin:$PATH"
#
# Install python packages
pdm install
#
# 
touch /nvcl-fs/testNFS.txt
# Run db update
cd src
pdm run make_reports.py -utb
EOF2
## Create script END

chmod +x batch_install.sh
chown batch: batch_install.sh

## Execute script as 'batch' user to allow NFS access
echo "Executing batch script"
su batch -c ./batch_install.sh
