# ARRUS on an NVIDIA IGX with us4OEM+ boards over Ethernet

Everything needed to take a fresh IGX Orin developer kit with a ConnectX NIC, two us4OEM+
boards on its two 40G ports, and end up running the ARRUS Python examples and the C++ bench
against them over the Ethernet bridge with the RDMA receiver. Validated on the reference
machine on 2026-09-16 (Ubuntu 22.04, kernel 5.15 tegra-igx, mlx5_core 23.10, RTX A6000).

Nothing here needs PCIe: the boards are reached through the us4OEM driver's Ethernet port
(the `us4r-api` repository, branch `ref-M_OEM-296`), and ARRUS drives them in the HOST work
mode exactly as it does over PCIe. Background, switches and measurements are in
[docs/dev/HostModeOverEthernet.md](../docs/dev/HostModeOverEthernet.md).

## 0. What you need

- The IGX with its ConnectX ports cabled to the boards' QSFP ports (passive DACs are fine),
  docker with the NVIDIA container toolkit, `rdma-core`. `igx/check-host.sh` verifies all of it.
- The two us4OEM+ boards, powered, carrying an Ethernet-bridge bitstream (status bit 11 set:
  little-endian samples on the wire; the reference boards boot 0x0A5EC7B1 from their
  configuration flash) and addressed 192.168.0.2 (master, port 1) and 192.168.4.2 (slave,
  port 2). The master is the board whose trigger output is looped to every board's
  trigger input; the order in `US4R_ETH_DEVICES` must put it first.
- HV: the boards use their own internal HVPS; a probe is needed only for imaging examples.
  STANDARD/AFE58JD18 boards must have HV enabled or their pulsers fault and the sequencer
  silently stalls after one frame; the examples and the bench enable it.
- Access to the two repositories: `us4useu/arrus` (this one, branch `ref-M_OEM-296`) and
  `us4r-api` (branch `ref-M_OEM-296` as well; ask us4us for access).

## 1. Host

```
git clone -b ref-M_OEM-296 git@github.com:us4useu/arrus.git
git clone -b ref-M_OEM-296 git@github.com:us4useu/us4r-api.git   # next to arrus/, or set US4R_API_DIR
cd arrus
sudo igx/setup-network.sh <master_port_if> <slave_port_if>   # 192.168.0.101 and 192.168.4.101, MTU 4096
igx/check-host.sh                                        # must end with "host check: OK"
```

`setup-network.sh` creates two NetworkManager connections with static addresses on the boards'
subnets and MTU 4096 (the bridge sends 4096-byte RoCE packets), and raises `net.core.rmem_max`
for the optional software receiver. Find the port names with `ip -br link` (on the reference
machine `enP5p3s0f1np1` is port 1 and `enP5p3s0f0np0` port 2).

## 2. Images

```
igx/build-images.sh
```

Builds `us4r-build` from the us4r-api repository's `.docker/build/Dockerfile` (CUDA 11.7 devel,
gcc 9, cmake 3.21, conan 1.59, python 3.8, libibverbs), `us4r-build-py` on top of it with
SWIG, numpy, scipy and matplotlib (`igx/docker/Dockerfile.py`), and `us4r-build-py310` with a
Python 3.10 built from source for the native wheel of section 5b (`igx/docker/Dockerfile.py310`). The base image pulls the CUDA
devel image and builds CMake; allow tens of minutes.

## 3. Driver (us4r-api, Ethernet port)

```
igx/build-driver.sh
```

Builds the us4OEM driver with its Ethernet port into `igx/out/us4-install` (`lib64/`,
`include/`, `bin/`), the tree ARRUS links against, following the "Building" section of the
us4r-api README: conan with the repository's `.conan/linux_aarch64.profile` (gcc 9, C++11 ABI),
cmake with `-DUS4R_BUILD_KERNEL_SPACE_DRIVER=OFF` (the branch replaces the PCIe transport, there
is no kernel module), `make`, `cmake --install`. The first run builds Boost from source
(45 to 90 minutes, once). The installed library carries a build marker:
`strings igx/out/us4-install/lib64/libUs4OEM.so | grep US4R-ETH-BUILD`.

Which commit: branch `ref-M_OEM-296` of `git@github.com:us4useu/us4r-api.git` (the same name
as this ARRUS branch) at `8a977c1a` or later (marker `mirror-unless-bit12-2026-09-16dh`). The
branch is not tagged. Do not use the older remote branch `eth-holoscan`, an ancestor hundreds
of commits back.

Two facts about the driver that matter at run time: `US4R_ETH_ALLOW_WRITES=1` must be set by
any process that arms a board (the control client refuses every register write without it,
because the control plane is single-client and a stray write from a diagnostic can hang the
bridge until a reconfigure; the run scripts set it), and `US4R_ETH_DEVICES` lists the boards
master first, each `host` or `host:port`.

## 4. ARRUS

```
igx/build-arrus.sh            # first run ~1 h: conan builds boost and protobuf from source
igx/build-arrus.sh --rebuild  # after a source change
```

Runs conan and cmake in `us4r-build-py` with the C++11 ABI (`compiler.libcxx=libstdc++11`,
anything else leaves libarrus-core with unresolved boost/protobuf symbols), builds the C++
core, the bench examples and the Python wheel (which embeds libarrus-core and the driver
libraries), then builds the runtime image `us4r-arrus-runtime` with the wheel, cupy and tkinter
installed and prints an import check. Outputs:

- `igx/out/build-py3.8/arrus/core/throughput-test` and `minimal-acquisition`, the C++ bench binaries
- `igx/out/wheel/arrus-*.whl`, the wheel (one per Python version built)
- the `us4r-arrus-runtime` image

## 5. Run the examples

```
mkdir -p ~/work && cd ~/work
cp <arrus>/igx/config/us4r_sl1543_esaote3.prototxt us4r.prototxt    # your probe and adapter
cp <arrus>/api/python/examples/plane_wave_imaging.py .
<arrus>/igx/run-example.sh plane_wave_imaging.py
```

`run-example.sh` starts the runtime image with the host network, the GPU, the RDMA devices and
an unlimited memlock, mounts the current directory at `/work`, sets `US4R_ETH_DEVICES` and
`US4R_ETH_ALLOW_WRITES=1`, and passes your `DISPLAY` through. For a live window either log in
with `ssh -X` (a few fps: every frame crosses the X connection) or run on the IGX desktop
with `DISPLAY=:0` after `xhost +local:` (about 30 fps). Without a `DISPLAY` the container gets
the Agg backend and the examples' Display2D returns immediately.

The session config: `probe_id` and `adapter_id` name entries of the ARRUS default dictionary
(`arrus/cfg/default.dict`); the file has no DTGC entry because the examples set an analog TGC
curve, which ARRUS does not combine with DTGC. `nus4oems: 2`, the watchdog off, and the
`us4oemhvps` HV model are what the Ethernet boards need; keep them.

Examples known to run unchanged on the reference bench with an SL1543 on the esaote3
adapter: `custom_tx_rx_sequence.py`, `plane_wave_imaging.py` (38 fps of B-mode headless).
`eth_bench_channel_order_probe.py` is the data-order acceptance check (section 7).

## 5b. Or install natively on the host, without docker at run time

The wheel is self-contained (libarrus-core, the driver and boost inside, needing only
rdma-core's libibverbs and libnl from the host), so it can be installed straight into a venv on
the IGX's system Python 3.10. That takes the cp310 wheel, built once (here or on the reference
machine) with the Python 3.10 build image:

```
ARRUS_PY_VERSION=3.10 ARRUS_BUILD_IMAGE=us4r-build-py310 igx/build-arrus.sh   # -> igx/out/wheel/arrus-*-cp310-*.whl
igx/install-native.sh [wheel]        # venv ~/arrus-venv: wheel, cupy, CUDA runtime wheels; no root
. ~/arrus-venv/bin/activate
cd ~/work && python3 plane_wave_imaging.py
```

The IGX ships the CUDA driver but no toolkit, so `install-native.sh` takes cupy and the CUDA
runtime libraries from PyPI (pinned to 12.3, which the 535 driver runs through minor-version
compatibility) and makes the venv's activation export their loader path together with the
board addresses and `US4R_ETH_ALLOW_WRITES=1`. The venv sees the system packages, so the
examples' window uses the host's tkinter and matplotlib. A wheel from another machine works
as long as it is a cp310 aarch64 wheel: copy it and pass its path to `install-native.sh`.

To ship without any build on the new machine: copy the wheel (native route) or
`docker save us4r-arrus-runtime | gzip > us4r-arrus-runtime.tgz` and `docker load` it there
(container route); the host steps of section 1 still apply.

## 6. Run the bench

```
igx/run-bench.sh single 1024 8 8 10 1000                  # single-firing 1 MiB elements, rx 8, 10 s, PRI 1 ms
THROUGHPUT_NTX=16 igx/run-bench.sh pwi16 1024 4 4 30 1000  # 16-firing elements
THROUGHPUT_CHECK=1 igx/run-bench.sh check 1024 8 8 10 1000 # with the ramp content check
```

No probe is needed (`igx/config/us4r_eth_bench_synthetic64.prototxt`, a synthetic 64-element
probe, 32 channels per board). Reference numbers with the RDMA receiver: 937 fps at 1 MiB
single-firing (two boards), 60 fps for 16-firing elements (the PRI limit), content clean.
The bench's switches are documented at the top of `arrus/core/examples/ThroughputTest.cpp`.

## 7. Acceptance after any change to the boards, the driver or ARRUS

1. `igx/check-host.sh` ends with OK.
2. `THROUGHPUT_CHECK=1 igx/run-bench.sh check 1024 8 8 10 1000`: no STALL, content clean, and
   the rate within about 10 % of the reference above.
3. With a probe: `TXEL=5 igx/run-example.sh eth_bench_channel_order_probe.py` prints
   `channel order: CORRECT`, and again with `TXEL=40`. The ramp check cannot see a channel
   permutation; this can.
4. `plane_wave_imaging.py` shows an image; without a display,
   `igx/run-example.sh <arrus>/igx/examples/plane_wave_imaging_headless.py` prints the frame
   rate (38 fps on the reference bench) and saves `plane_wave_imaging.png`.

## 8. Rules that cost a day each to learn

- Never rebuild or reinstall the driver or ARRUS while a session has the libraries mapped.
- One control client per board at a time; a second session on the same board wedges its
  control plane until both are closed.
- A power cycle is a cold boot: the boards come up from flash (since 2026-09-16) with the LMK
  reset, and the next session initialises them; it does not "reset" a wedged control plane
  faster than closing every client does.
- The software receiver (`US4R_ETH_RECEIVER=software`) needs host busy polling
  (`net.core.busy_read=50` as root, or `US4R_ETH_BUSY_POLL_US=50` with `CAP_NET_ADMIN`);
  without it the NIC holds the tail of a frame and a HOST-mode session deadlocks. The RDMA
  receiver, the default, does not have this problem.
- One firing per board must stay within 1 MiB (4096 samples x 128 channels x 2 bytes); a
  larger element is split over its firings by the driver, a larger single firing is refused.
- Two rx-buffer elements of a 32-angle, 4096-sample plane-wave sequence use 192 of the
  sequencer's 256 descriptor slots; deeper rx buffers on such sequences are refused at upload.
- When something stalls, read the driver's log lines before touching anything: the arm
  refusals name the missing device or setting, and ARRUS's "HOST releases at stop" line says
  whether the release path ran.

## Files

| path | what |
|---|---|
| `check-host.sh` | read-only host verification |
| `setup-network.sh` | the two port addresses, MTU, sysctl (sudo) |
| `build-images.sh` | `us4r-build`, `us4r-build-py` |
| `build-driver.sh` | us4r-api Ethernet port into `out/us4-install` |
| `build-arrus.sh` | core, bench, wheel (3.8 for the runtime image, 3.10 for the host), `us4r-arrus-runtime` |
| `install-native.sh` | the cp310 wheel into a host venv with cupy and the CUDA runtime wheels |
| `run-example.sh` | a Python script against the boards, with display |
| `run-bench.sh` | `throughput-test` against the boards |
| `config/us4r_sl1543_esaote3.prototxt` | session config for the SL1543 on esaote3 |
| `config/us4r_eth_bench_synthetic64.prototxt` | session config for the bench, no probe |
| `examples/plane_wave_imaging_headless.py` | the plane-wave example without a window, saves a PNG |
| `docker/Dockerfile.py`, `docker/Dockerfile.py310`, `docker/Dockerfile.runtime` | the derived images |
| `out/` | build products, git-ignored |
