#   Copyright (C) 2026 WatCAG, University of Waterloo
#   SPDX-License-Identifier: BSD-3-Clause

"""Alveo (PCIe) cards on PYNQ 4 through the XRT native API.

PYNQ 4.0.0 drives embedded boards only: ``XrtDevice._probe_`` always returns
``[]`` and ``pyxrt`` is imported only when ``/proc/device-tree/chosen/pynq_board``
exists (``pynq.ps.ON_TARGET``), so an x86 host with an Alveo card has no device.
``XrtNativeDevice`` restores Alveo support over ``libpynq_xrt.so``
(``_xrt/pynq_xrt.cpp``, built by setup.py when XRT is installed): device,
buffers, one exclusive hw_context, and CU registers through ``xrt::ip``.
Kernels start through PYNQ's register path (``DefaultIP.start``), with
``ap_continue`` for ap_ctrl_chain kernels, so ap_ctrl_none and user-managed
kernels work; no ERT commands are used.

The xclbin is loaded from its own path, so (unlike ``XrtDevice._xrt_download``)
nothing is written into the current directory.
"""

import ctypes
import os
import warnings

from pynq.ps import CPU_ARCH_IS_x86, ON_TARGET

from .xrt_device import XrtDevice

SHIM_NAME = "libpynq_xrt.so"

# None: not loaded yet; False: unavailable (missing library or XRT runtime).
_shim = None


def _bind(lib):
    for f in ("pxrt_open", "pxrt_bo", "pxrt_bo_map", "pxrt_ip"):
        getattr(lib, f).restype = ctypes.c_void_p
    lib.pxrt_error.restype = ctypes.c_char_p
    lib.pxrt_count.restype = ctypes.c_uint
    lib.pxrt_bo_size.restype = ctypes.c_size_t
    lib.pxrt_bo_address.restype = ctypes.c_uint64
    lib.pxrt_open.argtypes = [ctypes.c_uint]
    lib.pxrt_bo.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    lib.pxrt_bo_sync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
    lib.pxrt_bo_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.pxrt_bo_read.argtypes = lib.pxrt_bo_write.argtypes
    lib.pxrt_ip_read.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    lib.pxrt_ip_write.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    for f in ("pxrt_close", "pxrt_name", "pxrt_load", "pxrt_bo_free", "pxrt_bo_map", "pxrt_bo_size",
              "pxrt_bo_address", "pxrt_ip", "pxrt_ip_free"):
        getattr(lib, f).argtypes = [ctypes.c_void_p] + {
            "pxrt_name": [ctypes.c_char_p, ctypes.c_size_t], "pxrt_load": [ctypes.c_char_p],
            "pxrt_ip": [ctypes.c_char_p]}.get(f, [])
    return lib


def shim_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), SHIM_NAME)


def load_shim():
    """The bound libpynq_xrt.so, or None when it is not built or XRT is not installed."""
    global _shim
    if _shim is None:
        path = shim_path()
        if not os.path.isfile(path):
            _shim = False
        else:
            try:
                _shim = _bind(ctypes.CDLL(path))
            except OSError as e:
                warnings.warn("{} present but not loadable: {}".format(path, e))
                _shim = False
    return _shim or None


def _check(ok):
    if not ok:
        raise RuntimeError("XRT: " + _shim.pxrt_error().decode())
    return ok


class XrtNativeDevice(XrtDevice):
    """Alveo card through recent XRT (libpynq_xrt.so over the XRT native API).

    The xclbin is loaded into one exclusive hardware context and every kernel is
    driven through its registers (xrt::ip): no ERT commands.
    """

    # XrtDevice keeps 200 (and probes nothing on PYNQ 4); priorities must be unique.
    _probe_priority_ = 210

    @classmethod
    def _probe_(cls):
        # Embedded boards (ON_TARGET) belong to EmbeddedDevice/VersalDevice.
        if ON_TARGET or not CPU_ARCH_IS_x86 or load_shim() is None:
            return []
        return [cls(i) for i in range(_shim.pxrt_count())]

    def __init__(self, index, tag="xrt{}"):
        # XrtDevice.__init__ opens the card through pyxrt, which PYNQ 4 imports only
        # on embedded targets; set up the same attributes over the shim instead.
        if load_shim() is None:
            raise RuntimeError("XrtNativeDevice needs {} (built when XRT is installed)".format(shim_path()))
        super(XrtDevice, self).__init__(tag.format(index))
        self.capabilities = {
            "REGISTER_RW": True,
            "CALLABLE": True,
        }
        self._index = index
        self.handle = _check(_shim.pxrt_open(index))
        name = ctypes.create_string_buffer(256)
        _check(_shim.pxrt_name(self.handle, name, 256) == 0)
        self._name = name.value.decode()
        self.contexts = dict()
        self.sysfs_path = None
        self.active_bos = []
        self._bo_cache = []
        self._streams = {}

    @property
    def name(self):
        return self._name

    @property
    def device_info(self):
        raise RuntimeError("device_info needs pyxrt; XrtNativeDevice reports only name")

    @property
    def clocks(self):
        return {}

    @property
    def sensors(self):
        return {}

    def flush(self, bo, offset, ptr, size):
        _check(_shim.pxrt_bo_sync(bo, 1, size, offset) == 0)

    def invalidate(self, bo, offset, ptr, size):
        _check(_shim.pxrt_bo_sync(bo, 0, size, offset) == 0)

    def allocate_bo(self, size, idx, cacheable):
        return _check(_shim.pxrt_bo(self.handle, size, idx))

    def free_bo(self, bo):
        _shim.pxrt_bo_free(bo)

    def buffer_write(self, bo, bo_offset, buf, buf_offset=0, count=-1):
        view = memoryview(buf).cast("B")
        view = view[buf_offset:] if count == -1 else view[buf_offset : buf_offset + count]
        ptr = (ctypes.c_char * len(view)).from_buffer(view)
        _check(_shim.pxrt_bo_write(bo, ptr, len(view), bo_offset) == 0)

    def buffer_read(self, bo, bo_offset, buf, buf_offset=0, count=-1):
        view = memoryview(buf).cast("B")
        if view.readonly:
            raise RuntimeError("Buffer not writable")
        view = view[buf_offset:] if count == -1 else view[buf_offset : buf_offset + count]
        ptr = (ctypes.c_char * len(view)).from_buffer(view)
        _check(_shim.pxrt_bo_read(bo, ptr, len(view), bo_offset) == 0)

    def map_bo(self, bo):
        ptr = _check(_shim.pxrt_bo_map(bo))
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char * _shim.pxrt_bo_size(bo)))[0]

    def get_device_address(self, bo):
        return _shim.pxrt_bo_address(bo)

    def close(self):
        self.free_bitstream()
        if self.handle:
            _shim.pxrt_close(self.handle)
        self.handle = None

    def _ip_at(self, address):
        for c in self.contexts.values():
            if c["base"] <= address < c["base"] + c["range"]:
                return c["ip"], address - c["base"]
        raise RuntimeError("No open compute unit at address 0x{:x}".format(address))

    def read_registers(self, address, length):
        ip, offset = self._ip_at(address)
        value = ctypes.c_uint32()
        data = b""
        for o in range(offset, offset + length, 4):
            _check(_shim.pxrt_ip_read(ip, o, ctypes.byref(value)) == 0)
            data += value.value.to_bytes(4, "little")
        return data[:length]

    def write_registers(self, address, data):
        ip, offset = self._ip_at(address)
        for i in range(0, len(data), 4):
            word = int.from_bytes(data[i : i + 4], "little")
            _check(_shim.pxrt_ip_write(ip, offset + i, word) == 0)

    def free_bitstream(self):
        for c in self.contexts.values():
            _shim.pxrt_ip_free(c["ip"])
        self.contexts = dict()

    def _xrt_download(self, data):
        raise RuntimeError("XrtNativeDevice loads an xclbin from its path (download), not from bytes")

    def download(self, bitstream, parser=None):
        self.free_bitstream()
        _check(_shim.pxrt_load(self.handle, os.fsencode(os.path.abspath(bitstream.bitfile_name))) == 0)
        super().post_download(bitstream, parser)

    def open_context(self, description, shared=True):
        """Reserve the compute unit and map its registers (exclusive)"""
        cu_name = description["cu_name"]
        if cu_name not in self.contexts:
            kernel, _, inst = cu_name.partition(":")
            ip = _check(_shim.pxrt_ip(self.handle, "{}:{{{}}}".format(kernel, inst).encode()))
            self.contexts[cu_name] = {
                "ip": ip,
                "idx": description["cu_index"],
                "base": description["phys_addr"],
                "range": description["addr_range"],
            }
        return self.contexts[cu_name]["idx"]

    def close_context(self, cu_name):
        _shim.pxrt_ip_free(self.contexts.pop(cu_name)["ip"])

    def get_bitfile_metadata(self, bitfile_name, partial=False):
        from .xclbin_parser import XclBin

        return XclBin(bitfile_name)

    def set_event_loop(self, loop):
        raise RuntimeError("XrtNativeDevice has no XRT event interface; poll with WaitHandle.wait")
