#!/bin/bash

# Ubuntu version
UBUNTU_VER=${1}
OS=$(echo ${UBUNTU_VER} | tr -d '-' | tr -d '.')

# CUDA major and minor version
CUDA_VER_FULL=${2}
CUDA_VER_ARR=($(echo ${CUDA_VER_FULL} | tr "." " "))
CUDA_VER="${CUDA_VER_ARR[0]}.${CUDA_VER_ARR[1]}"
CUDA_VER_ID="${CUDA_VER_ARR[0]}_${CUDA_VER_ARR[1]}"
CUDA_VER_SHORT="cu${CUDA_VER_ARR[0]}${CUDA_VER_ARR[1]}"

case ${CUDA_VER_SHORT} in
  cu121)
    CUDA=12.1
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.1-530.30.02-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.1/local_installers
    ;;
  cu118)
    CUDA=11.8
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.0-520.61.05-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.0/local_installers
    ;;
  cu117)
    CUDA=11.7
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.1-515.65.01-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.1/local_installers
    ;;
  cu116)
    CUDA=11.6
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.2-510.47.03-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.2/local_installers
    ;;
  cu115)
    CUDA=11.5
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.2-495.29.05-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.2/local_installers
    ;;
  cu113)
    CUDA=11.3
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.0-465.19.01-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.0/local_installers
    ;;
  cu102)
    CUDA=10.2
    APT_KEY=${CUDA/./-}-local-${CUDA}.89-440.33.01
    FILENAME=cuda-repo-${OS}-${APT_KEY}_1.0-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}/Prod/local_installers
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=${CUDA_VER_SHORT}"
    exit 1
    ;;
esac

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -nv ${URL}/${FILENAME}
sudo dpkg -i ${FILENAME}

if [ "${CUDA_VER_SHORT}" = "cu117" ] || [ "${CUDA_VER_SHORT}" = "cu118" ] || [ "${CUDA_VER_SHORT}" = "cu121" ]; then
  sudo cp /var/cuda-repo-${APT_KEY}/cuda-*-keyring.gpg /usr/share/keyrings/
else
  sudo apt-key add /var/cuda-repo-${APT_KEY}/7fa2af80.pub
fi

sudo apt-get -qq update
sudo apt install -y cuda-nvcc-${CUDA/./-} cuda-libraries-dev-${CUDA/./-} cuda-command-line-tools-${CUDA/./-}
sudo apt clean

rm -f ${FILENAME}