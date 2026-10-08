# libpynq_xrt.so: XrtDevice's C entry points over the XRT native API (see pynq_xrt.cpp).
XILINX_XRT ?= /opt/xilinx/xrt

libpynq_xrt.so: pynq_xrt.cpp
	$(CXX) -O2 -std=c++17 -shared -fPIC -I$(XILINX_XRT)/include -o $@ $< \
	  -L$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_XRT)/lib -lxrt_coreutil

clean:
	rm -f libpynq_xrt.so
