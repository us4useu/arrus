#!/bin/bash
# Configures the two ConnectX ports for the boards, the way the reference IGX is set up:
# NetworkManager connections with static addresses on the boards' subnets and MTU 4096.
#   master board 192.168.0.2  <-  port 1  host 192.168.0.101/24
#   slave  board 192.168.4.2  <-  port 2  host 192.168.4.101/24
# Needs sudo. Idempotent (re-running updates the same connections).
#
#   sudo igx/setup-network.sh <master_port_if> <slave_port_if>     e.g. enP5p3s0f1np1 enP5p3s0f0np0
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: sudo $0 <master_port_if> <slave_port_if>"; exit 2; }
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
conf() { # ifname address
  local n="us4oem-$1"
  if nmcli -t -f NAME con show | grep -qx "$n"; then
    nmcli con mod "$n" ipv4.method manual ipv4.addresses "$2" ipv6.method disabled 802-3-ethernet.mtu 4096 connection.autoconnect yes
  else
    nmcli con add type ethernet ifname "$1" con-name "$n" ipv4.method manual ipv4.addresses "$2" ipv6.method disabled 802-3-ethernet.mtu 4096 connection.autoconnect yes
  fi
  nmcli con up "$n" > /dev/null
  echo "$1: $2 mtu 4096 ($n)"
}
conf "$1" 192.168.0.101/24
conf "$2" 192.168.4.101/24
# Socket receive buffer for the software receiver only (the RDMA receiver bypasses the kernel);
# harmless otherwise, and persistent via sysctl.d.
printf 'net.core.rmem_max = 33554432\n' > /etc/sysctl.d/90-us4oem-eth.conf && sysctl -q -p /etc/sysctl.d/90-us4oem-eth.conf
echo "done; check with igx/check-host.sh"
