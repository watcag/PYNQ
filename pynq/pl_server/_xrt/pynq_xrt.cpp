// C entry points over the XRT native API for PYNQ 3.0.1's XrtDevice (ctypes).  Recent XRT (2.23 checked) hides the
// xcl* shim functions PYNQ 3.0.1 binds (xclOpen, xclAllocBO, xclExecBuf, ...), and pyxrt has no register access (xrt::ip),
// which user-managed and ap_ctrl_none kernels need.  The xclbin is loaded into one exclusive hw_context: xocl
// maps a compute unit's registers for xrt::ip only when the CU is exclusively reserved.  PYNQ then drives every
// kernel through its registers (its software start path); no ERT commands are used.
// Errors: functions return null / nonzero and pxrt_error() gives the message.
//   built by setup.py (make -C pynq/pl_server/_xrt) when XRT is installed
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_ip.h>
#include <xrt/experimental/xrt_system.h>
#include <xrt/experimental/xrt_xclbin.h>

static thread_local std::string err;
#define TRY(fail, ...) try { __VA_ARGS__ } catch (const std::exception &e) { err = e.what(); return fail; }

struct Dev { xrt::device d; xrt::hw_context ctx; };

extern "C" {
const char *pxrt_error() { return err.c_str(); }
unsigned pxrt_count() { TRY(0u, return xrt::system::enumerate_devices();) }
void *pxrt_open(unsigned i) { TRY(nullptr, return new Dev{xrt::device(i), {}};) }
void pxrt_close(void *d) { delete static_cast<Dev *>(d); }
int pxrt_name(void *d, char *out, size_t n) {
  TRY(-1, std::strncpy(out, static_cast<Dev *>(d)->d.get_info<xrt::info::device::name>().c_str(), n - 1); return 0;)
}
int pxrt_load(void *d, const char *path) {  // drop any previous context (its ips must be closed), load, reserve
  TRY(-1, auto *D = static_cast<Dev *>(d); D->ctx = {};
      D->ctx = xrt::hw_context(D->d, D->d.register_xclbin(xrt::xclbin(std::string(path))),
                               xrt::hw_context::access_mode::exclusive);
      return 0;)
}
void *pxrt_bo(void *d, size_t size, unsigned group) {
  TRY(nullptr, return new xrt::bo(static_cast<Dev *>(d)->d, size, xrt::bo::flags::normal, group);)
}
void pxrt_bo_free(void *b) { delete static_cast<xrt::bo *>(b); }
void *pxrt_bo_map(void *b) { TRY(nullptr, return static_cast<xrt::bo *>(b)->map();) }
size_t pxrt_bo_size(void *b) { return static_cast<xrt::bo *>(b)->size(); }
uint64_t pxrt_bo_address(void *b) { return static_cast<xrt::bo *>(b)->address(); }
int pxrt_bo_sync(void *b, int to_device, size_t size, size_t offset) {
  TRY(-1, static_cast<xrt::bo *>(b)->sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE,
                                           size, offset); return 0;)
}
int pxrt_bo_write(void *b, const void *src, size_t size, size_t offset) {
  TRY(-1, static_cast<xrt::bo *>(b)->write(src, size, offset); return 0;)
}
int pxrt_bo_read(void *b, void *dst, size_t size, size_t offset) {
  TRY(-1, static_cast<xrt::bo *>(b)->read(dst, size, offset); return 0;)
}
void *pxrt_ip(void *d, const char *name) {  // name: "kernel:{instance}"
  TRY(nullptr, return new xrt::ip(static_cast<Dev *>(d)->ctx, std::string(name));)
}
void pxrt_ip_free(void *ip) { delete static_cast<xrt::ip *>(ip); }
int pxrt_ip_read(void *ip, uint32_t offset, uint32_t *value) {
  TRY(-1, *value = static_cast<xrt::ip *>(ip)->read_register(offset); return 0;)
}
int pxrt_ip_write(void *ip, uint32_t offset, uint32_t value) {
  TRY(-1, static_cast<xrt::ip *>(ip)->write_register(offset, value); return 0;)
}
}
