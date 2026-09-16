# HOST work mode over Ethernet (us4OEM+ Ethernet bridge)

Status 2026-09-16. This note records how ARRUS drives the HOST work mode when the us4OEM+
boards are reached over Ethernet instead of PCIe, what the host must provide, which
environment switches exist, and what was measured on the two-board bench (master 192.168.0.2,
slave 192.168.4.2, NVIDIA IGX host with a ConnectX NIC). The transport itself (ECB control
plane, bridge pages, RDMA and software receivers) lives in the us4OEM driver library; this
note covers the ARRUS side and the operational rules that follow from the driver's behaviour.

## Receiver: RDMA ships (decision 2026-09-16)

The us4OEM driver offers two host receivers for the transfer-to-host stream:

- **RDMA (verbs), the default.** `US4R_ETH_RECEIVER` unset selects it. The bridge addresses
  RoCEv2 UDP port 4791 and the NIC's RDMA engine places frames straight into the host buffer,
  bypassing the kernel receive ring. The host needs `rdma-core`, `/dev/infiniband/uverbs*`
  and `/dev/infiniband/rdma_cm`, and an unlimited memlock limit. In a container:
  `--device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 --device /dev/infiniband/rdma_cm --ulimit memlock=-1`.
  An absent or unusable verbs device refuses the arm with
  `RDMA receiver Start FAILED for <board>: <cause>`; there is no silent fallback.
- **Software receiver** (`US4R_ETH_RECEIVER=software`), a UDP socket. It is only supported
  with host busy polling: `net.core.busy_read=50` (and `net.core.busy_poll=50`) as a host
  sysctl, or `US4R_ETH_BUSY_POLL_US=50` when the process has `CAP_NET_ADMIN`. Without busy
  polling the mlx5 driver occasionally holds the tail of a frame in its receive ring until
  further traffic arrives on that ring. In HOST mode no further traffic comes, because the
  board is parked waiting for the release that waits for the completion, so the session
  deadlocks (about one run in ten on the bench, always within the first few dozen frames).
  The driver logs a WARNING at arm when the software receiver runs without busy polling.

The ring hold was pinned with an external instrument: the NIC's PHY 1024-1518 byte counter
advanced by exactly frames x 64 packets before every stall and by nothing during it, while
the ARRUS log shows the master's last completion arriving only when the stop's control
traffic flushed the ring. The BLOCK_CLR release path is not involved; sleeps before or after
the strobe only change the rate.

## What HOST mode does over Ethernet

ARRUS programs the same scheme as mainline v0.14.x does over PCIe:

- the last sequencer entry of every buffer element on every board carries a WAIT_FOR_SOFT
  park (`ARRUS_HOST_PARK` unset, i.e. `element`), no HS1/HS2 stop bits; with one firing per
  element that is every entry;
- one host buffer element = one bridge page region = one transfer index; when the element
  completes on every board the release callback strobes BLOCK_CLR on the **master only**
  (`Us4RImpl::syncTriggerAllOEMs()`); the slave's trigger input is gated by HW_TRIGGER_EN and
  runs one entry past its park by construction;
- the release also clears the entry's receive handshake range on every board
  (`MarkEntriesAsReadyForReceive`), as mainline does.

A parked master reads STATUS with CURRENT_INDEX = LAST_INDEX + 1 and BUSY = 1 (debug state
code 12); a parked slave reads BUSY = 0 with CURRENT_INDEX = LAST_INDEX.

## Buffer sizing

The Ethernet transfer ring is the host ring: the bridge writes page k of the ring for
transfer index k. A host buffer deeper than the rx buffer is allowed on driver builds that
route completions per entry (`IUs4OEM::TransferRingDepthIsHostRing()` returns false); on
older builds `Us4RImpl::upload()` refuses `hostBufferSize != rxBufferSize` with an
IllegalArgumentException naming both numbers, rather than letting the session stall.

## Environment switches (ARRUS side)

All are read at `upload()`/`start()`; unset means the default.

| Variable | Default | Meaning |
|---|---|---|
| `ARRUS_HOST_PARK` | `element` | `last` = one park per lap on the last entry plus HS stop bits (the PCIe-style scheme kept for comparison; not the shipping default). |
| `ARRUS_HOST_LAP_RELEASE` | off | `1` = with `ARRUS_HOST_PARK=last`, release once per lap from the last ring element. |
| `ARRUS_HOST_STALL_MS` | off | `N` = a watchdog releases the oldest incomplete element as LOST after N ms without a completion (clamped to at least four element periods; refused when the host buffer repeats). Off unless set. |
| `ARRUS_HOST_HS_RESUME` | `2` | HS resume strobe policy under `ARRUS_HOST_PARK=last` (2 counter-gated, 1 ungated, 0 off). No effect on the default scheme. |
| `ARRUS_HOST_NO_STOP_EN` | off | Diagnostic: leave the stop bits off even under `ARRUS_HOST_PARK=last`. |
| `ARRUS_SYNC_ALL_OEMS` | off | Diagnostic: strobe every board (slaves first) instead of the master only. |
| `ARRUS_SYNC_MASTER_ONLY`, `ARRUS_SYNC_PARK_DELAY_US`, `ARRUS_SYNC_SKIP_RELEASE_CLR` | off | SYNC-mode diagnostics, see `Us4RImpl.h`. |
| `ARRUS_EGRESS_ONLY_FIRING` | off | Diagnostic: request egress for one firing only (`Us4OEMDataTransferRegistrar.h`). |

`start()` logs one INFO line with the effective values. `stopDevice()` logs
`HOST releases at stop: N callback(s), N strobe(s) sent, N gated by state; last release T ms
before this stop; completed N element(s).` A stalled session with strobes = callbacks and a
last release seconds before the stop is a transport that never completed the next element; a
session with callbacks < completed elements is a release that did not run.

## Measured on the bench (2026-09-16, two boards, 1 ms PRI, content-checked)

| Shape | RDMA receiver | Software receiver, no busy poll | Software receiver, busy poll 50 us |
|---|---|---|---|
| single firing, 1 MiB elements, rx 8 | 937 fps, 0 stalls / 20 x 10 s | 686 fps, 2 stalls / 20 runs on plain defaults (6 / 60 over the release-delay variants) | 979 fps, 0 / 20 |
| 16 firings per element, rx 4 | 60 fps (PRI-bound), 125 MB/s, 3 x 30 s clean | | |
| single firing, rx 4, host buffer 8 | 695 to 742 fps, 3 x 10 s clean | | |
| plane_wave_imaging.py, SL1543, 32 angles x 4096 samples (24 MiB per element per board, 96 chained descriptors) | 38 fps headless B-mode | | |
| 16 firings, rx 8, 1 h soak (2026-09-15) | 214,934 elements, 0 lost, 0 stalls; 2149/2150 checked elements clean, the one failure undetailed | | |

The Python `custom_tx_rx_sequence.py` example runs over Ethernet with the RDMA receiver (a
copy adapted to the bench probe configuration, 5 frames in 0.21 s on 2026-09-16).

## Data-order probe (acceptance item since 2026-09-16)

The ramp test pattern verifies every channel's samples but is blind to any permutation of the
channels: a mirrored row order passed the 09-06 "ramp native LE 32/32" check and was found only
on a live image with a probe. The check that sees a permutation is
`api/python/examples/eth_bench_channel_order_probe.py`: fire one element with the full receive
aperture and report which column carries the transmit ringdown. On a correct data path the
strongest live column is the firing element; a mirrored row puts it at 31 - (k mod 32) inside
its 32-channel group. Run it after the ramp check on any change to the RX data path, the
driver's RX mapping write, or the bitstream. Background: the RTL's Data_Receiver indexed the
output lanes with `mapping[31-i]`, which the PCIe DMA's 512-bit endianness reversal used to
cancel; the fabric-side byte swap of 2026-09-06 (status bit 11) kept the swap and dropped the
cancellation. The driver mirrors the RX mapping table on bit-11 images without bit 12; a fixed
image announces native row order with status bit 12.

## Bench

`arrus/core/examples/ThroughputTest.cpp` (target `throughput-test`) drives the HOST scheme at a
fixed PRI for a fixed time, counts and content-checks elements, and prints the sequencer and
receiver state on a stall. `THROUGHPUT_MODE=HOST` is the default; `THROUGHPUT_NTX=N` sets the
firings per element, `THROUGHPUT_CHECK=1` the ramp content check. Usage and the other
switches are documented at the top of the file.

## Operational rules learned the hard way

- Never rebuild or reinstall the driver library while a session has it mapped.
- One ECB client per board at a time; two sessions on one board wedge the control plane.
- STANDARD/JD18 boards need HV enabled or the pulsers fault after one frame.
- Do not power-cycle the boards: the FPGA bitstream is volatile (JTAG).
- A measurement needs an instrument outside the path under test: the driver's counters and a
  packet sniffer sit behind the same receive ring; the PHY counters do not.
