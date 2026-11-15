# Copyright (C) 2026 WatCAG, University of Waterloo
# SPDX-License-Identifier: BSD-3-Clause

"""XrtNativeDevice (Alveo over libpynq_xrt.so) against an in-process fake of the shim.

No XRT, card or libpynq_xrt.so is needed: the fake implements every pxrt_* entry
point with ctypes buffers, so buffers are real host memory and registers a dict.
"""

import ctypes
import importlib
import os
import types

import numpy as np
import pytest

os.environ.setdefault("XILINX_XRT", "/opt/xilinx/xrt")

import pynq  # noqa: E402
import pynq.pl_server  # noqa: E402
from pynq.overlay import DefaultIP  # noqa: E402

xnd = importlib.import_module("pynq.pl_server.xrt_native_device")


class FakeShim:
    """The pxrt_* C ABI of pynq/pl_server/_xrt/pynq_xrt.cpp, in Python."""

    def __init__(self, count=1, name=b"xilinx_u280_gen3x16_xdma_base_1"):
        self.count, self.devname = count, name
        self.bos, self.ips, self.loads, self.syncs, self.freed = {}, {}, [], [], []
        self.next = 0x100
        self.error = b""
        self.ap_done_after = 0          # reads of CTRL before ap_done shows
        self.fail_load = False

    def _id(self):
        self.next += 0x10
        return self.next

    def pxrt_error(self):
        return self.error

    def pxrt_count(self):
        return self.count

    def pxrt_open(self, index):
        return self._id() if index < self.count else None

    def pxrt_close(self, handle):
        self.freed.append(("device", handle))

    def pxrt_name(self, handle, out, size):
        out.value = self.devname[: size - 1]
        return 0

    def pxrt_load(self, handle, path):
        if self.fail_load:
            self.error = b"load failed"
            return -1
        self.loads.append((path, os.getcwd(), sorted(os.listdir("."))))
        return 0

    def pxrt_bo(self, handle, size, group):
        bo = self._id()
        self.bos[bo] = {"mem": ctypes.create_string_buffer(size), "size": size, "group": group,
                        "address": 0x4000_0000_0000 + bo * 0x1_0000}
        return bo

    def pxrt_bo_free(self, bo):
        self.freed.append(("bo", bo))
        del self.bos[bo]

    def pxrt_bo_map(self, bo):
        return ctypes.addressof(self.bos[bo]["mem"])

    def pxrt_bo_size(self, bo):
        return self.bos[bo]["size"]

    def pxrt_bo_address(self, bo):
        return self.bos[bo]["address"]

    def pxrt_bo_sync(self, bo, to_device, size, offset):
        self.syncs.append((bo, "to" if to_device else "from", size, offset))
        return 0

    def pxrt_bo_write(self, bo, src, size, offset):
        ctypes.memmove(ctypes.addressof(self.bos[bo]["mem"]) + offset, src, size)
        return 0

    def pxrt_bo_read(self, bo, dst, size, offset):
        ctypes.memmove(dst, ctypes.addressof(self.bos[bo]["mem"]) + offset, size)
        return 0

    def pxrt_ip(self, handle, name):
        ip = self._id()
        self.ips[ip] = {"name": name, "regs": {}, "writes": [], "ctrl_reads": 0}
        return ip

    def pxrt_ip_free(self, ip):
        self.freed.append(("ip", ip))

    def pxrt_ip_read(self, ip, offset, value):
        cu = self.ips[ip]
        word = cu["regs"].get(offset, 0)
        if offset == 0:
            cu["ctrl_reads"] += 1
            if cu["ctrl_reads"] > self.ap_done_after and cu["regs"].get(0, 0) & 1:
                word = 0x4 | 0x2           # ap_done | ap_idle
        value._obj.value = word
        return 0

    def pxrt_ip_write(self, ip, offset, value):
        self.ips[ip]["writes"].append((offset, value))
        self.ips[ip]["regs"][offset] = value
        return 0


@pytest.fixture
def shim(monkeypatch):
    fake = FakeShim()
    monkeypatch.setattr(xnd, "_shim", fake)
    monkeypatch.setattr(xnd, "ON_TARGET", False)
    monkeypatch.setattr(xnd, "CPU_ARCH_IS_x86", True)
    return fake


@pytest.fixture
def device(shim):
    return xnd.XrtNativeDevice(0)


def ctrl_registers(chain=True):
    fields = {"AP_START": {}, "AP_DONE": {}, "AP_IDLE": {}, "AP_READY": {}}
    if chain:
        fields["AP_CONTINUE"] = {}
    return {
        "CTRL": {"address_offset": 0x00, "type": "unsigned int", "fields": fields,
                 "size": 32, "access": "read-write", "description": ""},
        "buf": {"address_offset": 0x10, "type": "void*", "size": 64, "access": "write-only",
                "description": "", "fields": {}},
        "n": {"address_offset": 0x1C, "type": "unsigned int", "size": 32, "access": "write-only",
              "description": "", "fields": {}},
    }


def cu_description(device, chain=True):
    return {"phys_addr": 0x1800000, "addr_range": 0x10000, "device": device,
            "registers": ctrl_registers(chain), "fullpath": "mm2s_0",
            "cu_name": "mm2s:mm2s_0", "cu_index": 3, "index": 3}


def test_probe_finds_cards_off_target(shim):
    shim.count = 2
    found = xnd.XrtNativeDevice._probe_()
    assert [d.name for d in found] == [shim.devname.decode()] * 2
    assert [d.tag for d in found] == ["xrt0", "xrt1"]
    assert all(d.has_capability("CALLABLE") and d.has_capability("REGISTER_RW") for d in found)
    assert not found[0].has_capability("MEMORY_MAPPED")


def test_probe_leaves_embedded_targets_alone(shim, monkeypatch):
    monkeypatch.setattr(xnd, "ON_TARGET", True)
    assert xnd.XrtNativeDevice._probe_() == []


def test_probe_without_shim_finds_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(xnd, "_shim", None)
    monkeypatch.setattr(xnd, "ON_TARGET", False)
    monkeypatch.setattr(xnd, "CPU_ARCH_IS_x86", True)
    monkeypatch.setattr(xnd, "shim_path", lambda: str(tmp_path / "libpynq_xrt.so"))
    assert xnd.XrtNativeDevice._probe_() == []
    with pytest.raises(RuntimeError, match="libpynq_xrt.so"):
        xnd.XrtNativeDevice(0)


def test_registered_with_a_unique_probe_priority():
    from pynq.pl_server.device import DeviceMeta
    assert DeviceMeta._subclasses[xnd.XrtNativeDevice._probe_priority_] is xnd.XrtNativeDevice
    assert pynq.pl_server.XrtNativeDevice is xnd.XrtNativeDevice


def test_download_loads_by_path_and_writes_nothing_to_cwd(device, shim, tmp_path, monkeypatch):
    image = tmp_path / "image" / "k.xclbin"
    image.parent.mkdir()
    image.write_bytes(b"xclbin2\0")
    cwd = tmp_path / "sealed"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    old = shim.pxrt_ip(device.handle, b"old:{old}")
    device.contexts["old"] = {"ip": old}
    bitstream = types.SimpleNamespace(bitfile_name=os.path.relpath(image), partial=True)
    device.download(bitstream)
    assert shim.loads == [(os.fsencode(str(image)), str(cwd), [])]
    assert os.listdir(cwd) == []                    # PYNQ 4 XrtDevice writes loaded.xclbin here
    assert device.contexts == {} and ("ip", old) in shim.freed   # previous CUs released first


def test_failed_download_raises_with_the_xrt_message(device, shim, tmp_path):
    shim.fail_load = True
    with pytest.raises(RuntimeError, match="XRT: load failed"):
        device.download(types.SimpleNamespace(bitfile_name=str(tmp_path / "x.xclbin"), partial=True))


def test_buffers_sync_the_requested_window_and_free_through_the_shim(device, shim):
    memory = device.get_memory({"idx": 5, "size": 1 << 20, "base_address": 0, "streaming": False,
                                "tag": "HBM[5]", "used": True})
    buf = memory.allocate((16,), np.uint64)
    bo = buf.bo
    assert shim.bos[bo]["group"] == 5 and shim.bos[bo]["size"] == 128     # no 8 KiB floor
    assert buf.physical_address == shim.bos[bo]["address"]
    buf[:] = np.arange(16, dtype=np.uint64)
    assert ctypes.string_at(shim.pxrt_bo_map(bo), 16) == np.arange(2, dtype=np.uint64).tobytes()
    buf.sync_to_device()
    buf[4:8].sync_from_device()
    assert shim.syncs == [(bo, "to", 128, 0), (bo, "from", 32, 32)]
    buf.freebuffer()
    assert ("bo", bo) in shim.freed and bo not in shim.bos


def test_registers_go_to_the_owning_compute_unit(device, shim):
    ip = DefaultIP(cu_description(device))
    (handle, cu), = shim.ips.items()
    assert cu["name"] == b"mm2s:{mm2s_0}"
    ip.write(0x1C, 0xABCD)
    assert cu["writes"] == [(0x1C, 0xABCD)]
    cu["regs"][0x20] = 0x1234
    assert ip.read(0x20) == 0x1234
    with pytest.raises(RuntimeError, match="No open compute unit at address 0x0"):
        device.read_registers(0, 4)


@pytest.mark.parametrize("chain", [True, False])
def test_start_and_wait_through_registers(device, shim, chain):
    ip = DefaultIP(cu_description(device, chain))
    (_, cu), = shim.ips.items()
    memory = device.get_memory({"idx": 0, "size": 1 << 20, "base_address": 0, "streaming": False,
                                "tag": "HBM[0]", "used": True})
    buf = memory.allocate((8,), np.uint64)
    shim.ap_done_after = 3
    handle = ip.start(buf, 8)
    address = shim.bos[buf.bo]["address"]
    args = {o: v for o, v in cu["writes"][:-1]}
    assert args[0x10] == address & 0xFFFFFFFF and args[0x14] == address >> 32 and args[0x1C] == 8
    assert cu["writes"][-1] == (0, 1)                  # ap_start after the arguments
    handle.wait()
    assert cu["ctrl_reads"] > shim.ap_done_after
    assert (cu["writes"][-1] == (0, 0x10)) is chain     # ap_continue for ap_ctrl_chain only
    assert str(ip.signature) == "(buf: 'void*', n: 'unsigned int')"
    ip.call(buf, n=8)


def test_ert_wait_lists_are_refused(device, shim):
    ip = DefaultIP(cu_description(device))
    with pytest.raises(RuntimeError, match="ERT"):
        ip.start(0, 1, waitfor=[])


def test_non_callable_devices_keep_pynq4_behaviour(device, shim):
    device.capabilities["CALLABLE"] = False             # EmbeddedDevice sets this
    ip = DefaultIP(cu_description(device))
    assert not hasattr(ip, "start") and not hasattr(ip, "call")


def test_close_releases_contexts_and_device(device, shim):
    DefaultIP(cu_description(device))
    handle = device.handle
    device.close()
    assert device.handle is None and device.contexts == {}
    assert ("device", handle) in shim.freed and any(kind == "ip" for kind, _ in shim.freed)


def test_bind_sets_the_c_abi():
    lib = types.SimpleNamespace(**{name: types.SimpleNamespace() for name in (
        "pxrt_error pxrt_count pxrt_open pxrt_close pxrt_name pxrt_load pxrt_bo pxrt_bo_free "
        "pxrt_bo_map pxrt_bo_size pxrt_bo_address pxrt_bo_sync pxrt_bo_write pxrt_bo_read "
        "pxrt_ip pxrt_ip_free pxrt_ip_read pxrt_ip_write").split()})
    xnd._bind(lib)
    assert lib.pxrt_open.restype is ctypes.c_void_p and lib.pxrt_bo_address.restype is ctypes.c_uint64
    assert lib.pxrt_load.argtypes == [ctypes.c_void_p, ctypes.c_char_p]
    assert lib.pxrt_bo_sync.argtypes == [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
