#!/bin/bash
# Checks an IGX host for everything the Ethernet us4OEM+ setup needs. Read-only; prints one line
# per check. Run before the first build and whenever the boards stop answering.
#
#   igx/check-host.sh [master_if slave_if]     (default: the two mlx5 ports carrying 192.168.0.x / 192.168.4.x)
set -u
ok()   { printf "  ok    %s\n" "$*"; }
bad()  { printf "  FAIL  %s\n" "$*"; FAILS=$((FAILS+1)); }
warn() { printf "  warn  %s\n" "$*"; }
FAILS=0
echo "host: $(hostname) $(uname -r) $(lsb_release -ds 2>/dev/null)"
command -v docker > /dev/null && ok "docker $(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" || bad "docker not installed"
docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia' && ok "nvidia container runtime" || bad "nvidia container runtime missing (nvidia-container-toolkit)"
CUDA_IMG=$(docker image inspect us4r-build > /dev/null 2>&1 && echo us4r-build || echo nvidia/cuda:11.7.1-base-ubuntu20.04)
docker run --rm --gpus all --entrypoint nvidia-smi "$CUDA_IMG" -L > /dev/null 2>&1 && ok "GPU visible in a container ($CUDA_IMG)" || warn "could not run a CUDA container with --gpus all ($CUDA_IMG; pulled if absent); the examples' GPU pipelines need it"
dpkg -s rdma-core > /dev/null 2>&1 && ok "rdma-core $(dpkg-query -W -f='${Version}' rdma-core)" || bad "rdma-core not installed"
for d in /dev/infiniband/uverbs0 /dev/infiniband/uverbs1 /dev/infiniband/rdma_cm; do [ -e $d ] && ok "$d" || bad "$d missing (mlx5 RDMA devices; is the ConnectX driver loaded?)"; done
command -v ibv_devices > /dev/null && ok "verbs devices: $(ibv_devices 2>/dev/null | awk 'NR>2{printf "%s ", $1}')" || warn "ibv_devices not found (ibverbs-utils); the driver opens the device itself"
[ "$(ulimit -l)" = "unlimited" ] || [ "$(ulimit -l)" -ge 1048576 ] 2>/dev/null && ok "memlock limit $(ulimit -l)" || warn "memlock limit $(ulimit -l) kB for this shell (the containers run with --ulimit memlock=-1, so this only matters outside docker)"
# NIC ports: by argument, or the mlx5 ports that carry the bench subnets.
if [ $# -ge 2 ]; then PORTS="$1 $2"; else PORTS=$(ip -br -4 addr | awk '$3 ~ /^192\.168\.(0|4)\./ {print $1}' | tr '\n' ' '); fi
[ -n "$PORTS" ] || bad "no interface carries 192.168.0.x or 192.168.4.x (see igx/README.md, host network)"
for p in $PORTS; do
  addr=$(ip -br -4 addr show dev $p | awk '{print $3}'); state=$(cat /sys/class/net/$p/operstate); speed=$(cat /sys/class/net/$p/speed 2>/dev/null); mtu=$(cat /sys/class/net/$p/mtu)
  drv=$(ethtool -i $p 2>/dev/null | awk '/^driver:/{print $2}')
  [ "$state" = up ] && ok "$p $addr link up, ${speed} Mb/s, mtu $mtu, driver $drv" || bad "$p $addr link $state"
  [ "$mtu" -ge 4096 ] || warn "$p mtu $mtu: the bridge sends 4096-byte RoCE packets, set mtu 4096 (or larger) on this port"
  [ "$drv" = mlx5_core ] || warn "$p driver $drv: the RDMA receiver was validated on mlx5 (ConnectX) only"
done
for b in $(echo "${US4R_ETH_DEVICES:-192.168.0.2,192.168.4.2}" | tr ',' ' '); do
  ping -c 1 -W 1 $b > /dev/null 2>&1 && ok "board $b answers ping" || bad "board $b does not answer ping"
done
for i in us4r-build us4r-build-py us4r-arrus-runtime; do docker image inspect $i > /dev/null 2>&1 && ok "image $i" || warn "image $i not built yet (igx/build-images.sh, igx/build-arrus.sh)"; done
[ $FAILS -eq 0 ] && echo "host check: OK" || { echo "host check: $FAILS failure(s)"; exit 1; }
