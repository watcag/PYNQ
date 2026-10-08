#   Copyright (c) 2019-2022, Xilinx, Inc.
#   SPDX-License-Identifier: BSD-3-Clause

import asyncio
import copy
import ctypes
import errno
import glob
import os
import subprocess
import warnings
import weakref

import numpy as np
from pynq._3rdparty import ert, xrt
from pynq.buffer import PynqBuffer
from pynq.ps import CPU_ARCH_IS_x86

from .device import Device



DRM_XOCL_BO_EXECBUF = 1 << 31
REQUIRED_VERSION_ERT = (2, 3, 0)
ZOCL_BO_FLAGS_CACHEABLE = 1 << 24
libc = ctypes.CDLL("libc.so.6")

libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

# Recent XRT (2.23 checked) no longer exports the xcl* functions of _3rdparty/xrt.py; libpynq_xrt.so
# (pynq_xrt.cpp, built next to this file) wraps the XRT native API instead.
_shim = None
if xrt.XRT_SUPPORTED and CPU_ARCH_IS_x86:
    _shim = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libpynq_xrt.so"))
    for _f in ("pxrt_open", "pxrt_bo", "pxrt_bo_map", "pxrt_ip"):
        getattr(_shim, _f).restype = ctypes.c_void_p
    _shim.pxrt_error.restype = ctypes.c_char_p
    _shim.pxrt_bo_size.restype = ctypes.c_size_t
    _shim.pxrt_bo_address.restype = ctypes.c_uint64
    _shim.pxrt_open.argtypes = [ctypes.c_uint]
    _shim.pxrt_bo.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    _shim.pxrt_bo_sync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
    _shim.pxrt_bo_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    _shim.pxrt_bo_read.argtypes = _shim.pxrt_bo_write.argtypes
    _shim.pxrt_ip_read.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    _shim.pxrt_ip_write.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    for _f in ("pxrt_close", "pxrt_name", "pxrt_load", "pxrt_bo_free", "pxrt_bo_map", "pxrt_bo_size",
               "pxrt_bo_address", "pxrt_ip", "pxrt_ip_free"):
        getattr(_shim, _f).argtypes = [ctypes.c_void_p] + {
            "pxrt_name": [ctypes.c_char_p, ctypes.c_size_t], "pxrt_load": [ctypes.c_char_p],
            "pxrt_ip": [ctypes.c_char_p]}.get(_f, [])


def _check(ok):
    if not ok:
        raise RuntimeError("XRT: " + _shim.pxrt_error().decode())
    return ok


# Create our own struct that fixes the typos of the real one
class xclDeviceUsage(ctypes.Structure):
    _fields_ = [
        ("h2c", ctypes.c_size_t * 8),
        ("c2h", ctypes.c_size_t * 8),
        ("ddrMemUsed", ctypes.c_size_t * 8),
        ("ddrBOAllocated", ctypes.c_uint * 8),
        ("totalContents", ctypes.c_uint),
        ("xclbinId", ctypes.c_ulonglong * 4),
        ("dma_channel_cnt", ctypes.c_uint),
        ("mm_channel_cnt", ctypes.c_uint),
        ("memSize", ctypes.c_ulonglong * 8),
    ]


_xrt_errors = {
    -95: "Shell does not match",
    -16: "Bitstream in use by another program",
    -1: "Possibly buffers still allocated",
}


def _get_xrt_version_embedded():
    output = subprocess.run(
        ["xbutil", "--version"], stdout=subprocess.PIPE, universal_newlines=True
    )
    if output.returncode != 0:
        warnings.warn("xbutil failed to run - unable to determine XRT version")
        return (0, 0, 0)
    xrt_version_str = output.stdout.split("\n")[0].split(":")[1].strip()
    return tuple(map(int, xrt_version_str.split(".")))


def _get_xrt_version_x86():
    import json

    try:
        with open(os.environ["XILINX_XRT"] + "/version.json", "r") as f:
            details = json.loads(f.read())
        return tuple(int(s) for s in details["BUILD_VERSION"].split("."))
    except Exception:
        warnings.warn("Unable to determine XRT version")
        return (0, 0, 0)


if xrt.XRT_SUPPORTED:
    _xrt_version = (
        _get_xrt_version_x86() if CPU_ARCH_IS_x86 else _get_xrt_version_embedded()
    )
else:
    _xrt_version = (0, 0, 0)


def _format_xrt_error(err):
    errstring = "{} ({}) {}".format(errno.errorcode[-err], -err, os.strerror(-err))
    if err in _xrt_errors:
        errstring += "/" + _xrt_errors[err]
    return errstring


def _xrt_allocate(shape, dtype, device, memidx, cacheable=0, pointer=None, cache=None):
    elements = 1
    try:
        for s in shape:
            elements *= s
    except TypeError:
        elements = shape
    dtype = np.dtype(dtype)
    size = elements * dtype.itemsize
    if pointer is not None:
        bo, buf, device_address = pointer
    else:
        bo = device.allocate_bo(size, memidx, cacheable)
        buf = device.map_bo(bo)
        device_address = device.get_device_address(bo)
    ar = PynqBuffer(
        shape,
        dtype,
        bo=bo,
        device=device,
        buffer=buf,
        device_address=device_address,
        coherent=False,
    )
    if pointer is not None:
        weakref.finalize(buf, _free_bo, device, bo, ar.virtual_address, ar.nbytes)
    if cache is not None:
        ar.pointer = (bo, buf, device_address)
        ar.return_to = cache
    return ar


def _free_bo(device, bo, ptr, length):
    device.free_bo(bo)


class XrtMemory:
    """Class representing a memory bank in a card

    Memory banks can be both external DDR banks and internal buffers.
    XrtMemory instances for the same bank are interchangeable and can
    be compared and used as dictionary keys.

    """

    def __init__(self, device, desc):
        self.idx = desc["idx"]
        self.size = desc["size"]
        self.base_address = desc["base_address"]
        self.desc = desc
        self.device = device

    def allocate(self, shape, dtype, **kwargs):
        """Create a new  buffer in the memory bank

        Parameters
        ----------
        shape : tuple(int)
            Shape of the array
        dtype : np.dtype
            Data type of the array

        """
        buf = _xrt_allocate(shape, dtype, self.device, self.idx, **kwargs)
        buf.memory = self
        return buf

    def __hash__(self):
        return hash((self.device, self.idx))

    def __eq__(self, other):
        return (
            type(other) is XrtMemory
            and self.device == other.device
            and self.idx == other.idx
        )

    @property
    def mem_used(self):
        usage = self.device.get_usage()
        return usage.ddrMemUsed[self.idx]

    @property
    def num_buffers(self):
        usage = self.device.get_usage()
        return usage.ddrBOAllocated[self.idx]


class XrtUUID:
    def __init__(self, val):
        self.bytes = val


class ExecBo:
    """Execution Buffer Object

    Wraps an execution buffer used by XRT to schedule the execution of
    accelerators. Usually used in conjunction with the ERT packet format
    exposed in the XRT ``ert_binding`` python module.

    """

    def __init__(self, bo, ptr, device, length):
        self.bo = bo
        self.ptr = ptr
        self.device = device
        self.length = length

    def __del__(self):
        _free_bo(self.device, self.bo, self.ptr, self.length)

    def as_packet(self, ptype):
        """Get a packet representation of the buffer object

        Parameters
        ----------
        ptype : ctypes struct
            The type to cast the buffer to

        """
        return ctypes.cast(self.ptr, ctypes.POINTER(ptype))[0]


class ErtWaitHandle:
    """WaitHandle specific to ERT-scheduled accelerators"""

    def __init__(self, bo, future, device):
        self._future = future
        self._bo = bo
        self.device = device

    def _complete(self, state):
        if state != ert.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            self._future.set_exception(RuntimeError("Execution failed: " + str(state)))
        else:
            self._future.set_result(None)
        self._bo = None

    @property
    def _has_bo(self):
        return self._bo is not None

    @property
    def done(self):
        """True is the accelerator has finished"""
        return self._future.done()

    async def wait_async(self):
        """Coroutine to wait for the execution to be completed

        This function requires that ``XrtDevice.set_event_loop`` is called
        before the accelerator execution is started

        """
        await self._future

    def wait(self):
        """Wait for the Execution to be completed"""
        while not self.done:
            self.device._handle_events(1000)


class XrtStream:
    """XRT Streaming Connection

    Encapsulates the IP connected to a stream. Note that the ``_ip``
    attributes will only be populated if the corresponding device
    driver has been instantiated.

    Attributes
    ----------
    source : str
        Source of the streaming connection as ip_name.port
    sink : str
        Sink of the streaming connection as ip_name.port
    monitors : [str]
        Monitor connections of the stream as a list of ip_name.port
    source_ip : pynq.overlay.DefaultIP
        Source IP driver instance for the stream
    sink_ip : pynq.overlay.DefaultIP
        Sink IP driver instance for the stream
    monitor_ips : [pynq.overlay.DefaultIP]
        list of driver instances for IP monitoring the stream

    """

    def __init__(self, device, desc):
        ip_dict = device.ip_dict
        idx = desc["idx"]
        for ip_name, ip in ip_dict.items():
            for stream_name, stream in ip["streams"].items():
                if stream["stream_id"] == idx:
                    if stream["direction"] == "output":
                        self.source = ip_name + "." + stream_name
                    elif stream["direction"] == "input":
                        self.sink = ip_name + "." + stream_name
        self.source_ip = None
        self.monitors = []
        self.monitor_ips = []
        self.sink_ip = None

    def __repr__(self):
        return "XrtStream(source={}, sink={})".format(self.source, self.sink)


class XrtDevice(Device):
    """Alveo card through recent XRT (libpynq_xrt.so over the XRT native API).

    The xclbin is loaded into one exclusive hardware context and every kernel is
    driven through its registers (xrt::ip): no ERT commands.
    """

    @classmethod
    def _probe_(cls):
        if not xrt.XRT_SUPPORTED or not CPU_ARCH_IS_x86:
            return []
        return [XrtDevice(i) for i in range(_shim.pxrt_count())]

    _probe_priority_ = 200

    def __init__(self, index, tag="xrt{}"):
        super().__init__(tag.format(index))
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
        self._loop = asyncio.get_event_loop()
        self._streams = {}

    @property
    def name(self):
        return self._name

    @property
    def clocks(self):
        return {}

    @property
    def default_memory(self):
        mem_dict = self.mem_dict
        active_mems = [m for m in mem_dict.values() if m["used"] and not m["streaming"]]
        if len(active_mems) == 0:
            raise RuntimeError("No active memories in design")
        elif len(active_mems) > 1:
            raise RuntimeError(
                "Multiple memories active in design: specify"
                + " the memory using the `target` parameters"
            )
        return self.get_memory(active_mems[0])

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

    def get_memory(self, desc):
        if desc["streaming"]:
            if desc["idx"] not in self._streams:
                self._streams[desc["idx"]] = XrtStream(self, desc)
            return self._streams[desc["idx"]]
        else:
            return XrtMemory(self, desc)

    def get_memory_by_idx(self, idx):
        for m in self.mem_dict.values():
            if m["idx"] == idx:
                return self.get_memory(m)
        raise RuntimeError("Could not find memory")

    def get_memory_by_name(self, name):
        for m in self.mem_dict.values():
            if m["tag"] == name:
                return self.get_memory(m)
        raise RuntimeError("Could not find memory")

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

    def gen_cache(self, bitstream, parser=None):
        pass

    def download(self, bitstream, parser=None):
        self.free_bitstream()
        _check(_shim.pxrt_load(self.handle, bitstream.bitfile_name.encode()) == 0)
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

    def get_bitfile_metadata(self, bitfile_name):
        from .xclbin_parser import XclBin

        return XclBin(bitfile_name)

    def get_exec_bo(self, size=1024):
        if len(self._bo_cache):
            return self._bo_cache.pop()
        if _xrt_version < REQUIRED_VERSION_ERT:
            raise RuntimeError("XRT Version too old for PYNQ ERT support")
        new_bo = xrt.xclAllocBO(self.handle, size, 0, DRM_XOCL_BO_EXECBUF)
        new_ptr = xrt.xclMapBO(self.handle, new_bo, 1)
        return ExecBo(new_bo, new_ptr, self, size)

    def return_exec_bo(self, bo):
        self._bo_cache.append(bo)

    def execute_bo(self, bo):
        status = xrt.xclExecBuf(self.handle, bo.bo)
        if status:
            raise RuntimeError("Buffer submit failed: " + str(status))
        wh = ErtWaitHandle(bo, self._loop.create_future(), self)
        self.active_bos.append((bo, wh))
        return wh

    def execute_bo_with_waitlist(self, bo, waitlist):
        if _xrt_version >= (2, 11, 0):
            raise RuntimeError(
                "waitfor list to schedule dependent executions "
                "is not supported by XRT anymore."
            )
        wait_array = (ctypes.c_uint * len(waitlist))()
        for i in range(len(waitlist)):
            wait_array[i] = waitlist[i].bo
        status = xrt.xclExecBufWithWaitList(
            self.handle, bo.bo, len(waitlist), wait_array
        )
        if status:
            raise RuntimeError("Buffer submit failed: " + str(status))
        wh = ErtWaitHandle(bo, self._loop.create_future(), self)
        self.active_bos.append((bo, wh))
        return wh

    def set_event_loop(self, loop):
        self._loop = loop
        for fd in glob.glob("/proc/self/fd/*"):
            try:
                link_target = os.readlink(fd)
            except:
                continue
            if link_target.startswith("/dev/dri/renderD"):
                base_fd = int(os.path.basename(fd))
                loop.add_reader(open(base_fd, closefd=False), self._handle_events)

    def _handle_events(self, timeout=0):
        xrt.xclExecWait(self.handle, timeout)
        next_bos = []
        for bo, completion in self.active_bos:
            state = bo.as_packet(ert.ert_cmd_struct).state & 0xF
            if state >= ert.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                if completion:
                    completion._complete(state)
                self.return_exec_bo(bo)
            else:
                next_bos.append((bo, completion))
        self.active_bos = next_bos


