//! Audited, dependency-light CUDA Driver API surface for prebuilt dispatch.
//!
//! This crate never links a CUDA runtime and never compiles device code.  It
//! loads the NVIDIA driver's stable C ABI at runtime, retains one primary
//! context, and owns every stream/module/allocation through RAII.  Unsafe code
//! is confined to symbol loading and those FFI calls; callers receive a safe,
//! typed launch and copy surface.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::fmt;
use std::ptr;
use std::sync::Arc;

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
pub type DevicePtr = u64;

const CUDA_SUCCESS: CuResult = 0;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_STREAM_NON_BLOCKING: c_uint = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaError {
    pub call: String,
    pub status: i32,
    pub name: String,
    pub detail: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} failed: {} ({}): {}",
            self.call, self.name, self.status, self.detail
        )
    }
}

impl std::error::Error for CudaError {}

#[derive(Debug, Clone, Copy)]
pub enum KernelArg {
    Ptr(DevicePtr),
    I32(i32),
    U32(u32),
    Bool(bool),
}

enum ArgStorage {
    Ptr(DevicePtr),
    I32(i32),
    U32(u32),
    Bool(u8),
}

impl ArgStorage {
    fn address(&self) -> *mut c_void {
        match self {
            Self::Ptr(value) => ptr::from_ref(value).cast_mut().cast(),
            Self::I32(value) => ptr::from_ref(value).cast_mut().cast(),
            Self::U32(value) => ptr::from_ref(value).cast_mut().cast(),
            Self::Bool(value) => ptr::from_ref(value).cast_mut().cast(),
        }
    }
}

impl From<KernelArg> for ArgStorage {
    fn from(value: KernelArg) -> Self {
        match value {
            KernelArg::Ptr(value) => Self::Ptr(value),
            KernelArg::I32(value) => Self::I32(value),
            KernelArg::U32(value) => Self::U32(value),
            KernelArg::Bool(value) => Self::Bool(u8::from(value)),
        }
    }
}

type CuGetErrorName = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;
type CuGetErrorString = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;
type CuInit = unsafe extern "C" fn(c_uint) -> CuResult;
type CuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult;
type CuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;
type CuDevicePrimaryCtxRelease = unsafe extern "C" fn(CuDevice) -> CuResult;
type CuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
type CuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type CuMemAlloc = unsafe extern "C" fn(*mut DevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "C" fn(DevicePtr) -> CuResult;
type CuMemcpyHtoDAsync =
    unsafe extern "C" fn(DevicePtr, *const c_void, usize, CuStream) -> CuResult;
type CuMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, DevicePtr, usize, CuStream) -> CuResult;
type CuMemsetD8Async = unsafe extern "C" fn(DevicePtr, u8, usize, CuStream) -> CuResult;
type CuMemsetD32Async = unsafe extern "C" fn(DevicePtr, u32, usize, CuStream) -> CuResult;
type CuStreamCreate = unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult;
type CuStreamDestroy = unsafe extern "C" fn(CuStream) -> CuResult;
type CuStreamSynchronize = unsafe extern "C" fn(CuStream) -> CuResult;
type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CuStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;

struct Api {
    library: *mut c_void,
    get_error_name: CuGetErrorName,
    get_error_string: CuGetErrorString,
    init: CuInit,
    driver_get_version: CuDriverGetVersion,
    device_get: CuDeviceGet,
    device_get_attribute: CuDeviceGetAttribute,
    primary_retain: CuDevicePrimaryCtxRetain,
    primary_release: CuDevicePrimaryCtxRelease,
    ctx_set_current: CuCtxSetCurrent,
    module_load_data: CuModuleLoadData,
    module_unload: CuModuleUnload,
    module_get_function: CuModuleGetFunction,
    mem_alloc: CuMemAlloc,
    mem_free: CuMemFree,
    memcpy_htod_async: CuMemcpyHtoDAsync,
    memcpy_dtoh_async: CuMemcpyDtoHAsync,
    memset_d8_async: CuMemsetD8Async,
    memset_d32_async: CuMemsetD32Async,
    stream_create: CuStreamCreate,
    stream_destroy: CuStreamDestroy,
    stream_synchronize: CuStreamSynchronize,
    launch_kernel: CuLaunchKernel,
}

// The driver API is process-global and thread-safe. Context selection remains
// explicit before every operation that consumes a context-owned handle.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    fn load() -> Result<Arc<Self>, CudaError> {
        let mut library = ptr::null_mut();
        for candidate in ["libcuda.so.1", "libcuda.so"] {
            let name = CString::new(candidate).expect("literal");
            // SAFETY: name is NUL-terminated; the returned handle is retained
            // by `Api` until every copied function pointer is unreachable.
            library = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !library.is_null() {
                break;
            }
        }
        if library.is_null() {
            return Err(CudaError {
                call: "dlopen(libcuda)".to_owned(),
                status: -1,
                name: "CUDA_ERROR_NOT_FOUND".to_owned(),
                detail: "libcuda.so.1 is not present".to_owned(),
            });
        }

        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                let name = CString::new($name).expect("literal");
                // SAFETY: the handle is live and the requested symbol is
                // copied into the exact CUDA C ABI signature declared above.
                let raw = unsafe { libc::dlsym(library, name.as_ptr()) };
                if raw.is_null() {
                    // SAFETY: this is the handle obtained above and no copied
                    // function pointer has escaped on this error path.
                    unsafe { libc::dlclose(library) };
                    return Err(CudaError {
                        call: format!("dlsym({})", $name),
                        status: -1,
                        name: "CUDA_ERROR_SYMBOL_NOT_FOUND".to_owned(),
                        detail: format!("the NVIDIA driver exports no {}", $name),
                    });
                }
                // SAFETY: CUDA documents this symbol with `$ty`'s ABI.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(raw) }
            }};
        }

        let api = Arc::new(Self {
            library,
            get_error_name: symbol!("cuGetErrorName", CuGetErrorName),
            get_error_string: symbol!("cuGetErrorString", CuGetErrorString),
            init: symbol!("cuInit", CuInit),
            driver_get_version: symbol!("cuDriverGetVersion", CuDriverGetVersion),
            device_get: symbol!("cuDeviceGet", CuDeviceGet),
            device_get_attribute: symbol!("cuDeviceGetAttribute", CuDeviceGetAttribute),
            primary_retain: symbol!("cuDevicePrimaryCtxRetain", CuDevicePrimaryCtxRetain),
            primary_release: symbol!("cuDevicePrimaryCtxRelease_v2", CuDevicePrimaryCtxRelease),
            ctx_set_current: symbol!("cuCtxSetCurrent", CuCtxSetCurrent),
            module_load_data: symbol!("cuModuleLoadData", CuModuleLoadData),
            module_unload: symbol!("cuModuleUnload", CuModuleUnload),
            module_get_function: symbol!("cuModuleGetFunction", CuModuleGetFunction),
            mem_alloc: symbol!("cuMemAlloc_v2", CuMemAlloc),
            mem_free: symbol!("cuMemFree_v2", CuMemFree),
            memcpy_htod_async: symbol!("cuMemcpyHtoDAsync_v2", CuMemcpyHtoDAsync),
            memcpy_dtoh_async: symbol!("cuMemcpyDtoHAsync_v2", CuMemcpyDtoHAsync),
            memset_d8_async: symbol!("cuMemsetD8Async", CuMemsetD8Async),
            memset_d32_async: symbol!("cuMemsetD32Async", CuMemsetD32Async),
            stream_create: symbol!("cuStreamCreate", CuStreamCreate),
            stream_destroy: symbol!("cuStreamDestroy_v2", CuStreamDestroy),
            stream_synchronize: symbol!("cuStreamSynchronize", CuStreamSynchronize),
            launch_kernel: symbol!("cuLaunchKernel", CuLaunchKernel),
        });
        api.check(unsafe { (api.init)(0) }, "cuInit")?;
        Ok(api)
    }

    fn error_text(&self, status: CuResult) -> (String, String) {
        let mut name = ptr::null();
        let mut detail = ptr::null();
        // SAFETY: the driver writes borrowed pointers to static strings.
        unsafe {
            (self.get_error_name)(status, &mut name);
            (self.get_error_string)(status, &mut detail);
        }
        let read = |raw: *const c_char, fallback: &str| {
            if raw.is_null() {
                fallback.to_owned()
            } else {
                // SAFETY: CUDA promises a NUL-terminated error string.
                unsafe { CStr::from_ptr(raw) }
                    .to_string_lossy()
                    .into_owned()
            }
        };
        (
            read(name, "CUDA_ERROR_UNKNOWN"),
            read(detail, "unknown CUDA error"),
        )
    }

    fn check(&self, status: CuResult, call: impl Into<String>) -> Result<(), CudaError> {
        if status == CUDA_SUCCESS {
            return Ok(());
        }
        let (name, detail) = self.error_text(status);
        Err(CudaError {
            call: call.into(),
            status,
            name,
            detail,
        })
    }
}

impl Drop for Api {
    fn drop(&mut self) {
        if !self.library.is_null() {
            // SAFETY: every owner of a copied function pointer owns this Api.
            unsafe { libc::dlclose(self.library) };
        }
    }
}

struct ContextInner {
    api: Arc<Api>,
    device: CuDevice,
    context: CuContext,
}

unsafe impl Send for ContextInner {}
unsafe impl Sync for ContextInner {}

impl ContextInner {
    fn current(&self) -> Result<(), CudaError> {
        // SAFETY: context is retained for this object's lifetime.
        self.api.check(
            unsafe { (self.api.ctx_set_current)(self.context) },
            "cuCtxSetCurrent",
        )
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        let _ = self.current();
        // SAFETY: this balances the successful retain in `CudaContext::new`.
        unsafe { (self.api.primary_release)(self.device) };
    }
}

#[derive(Clone)]
pub struct CudaContext {
    inner: Arc<ContextInner>,
    driver_version: i32,
    architecture: (i32, i32),
}

impl fmt::Debug for CudaContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaContext")
            .field("driver_version", &self.driver_version)
            .field("architecture", &self.architecture)
            .finish_non_exhaustive()
    }
}

impl CudaContext {
    pub fn new(ordinal: i32) -> Result<Self, CudaError> {
        let api = Api::load()?;
        let mut device = 0;
        api.check(
            unsafe { (api.device_get)(&mut device, ordinal) },
            "cuDeviceGet",
        )?;
        let mut context = ptr::null_mut();
        api.check(
            unsafe { (api.primary_retain)(&mut context, device) },
            "cuDevicePrimaryCtxRetain",
        )?;
        let inner = Arc::new(ContextInner {
            api,
            device,
            context,
        });
        inner.current()?;
        let mut driver_version = 0;
        inner.api.check(
            unsafe { (inner.api.driver_get_version)(&mut driver_version) },
            "cuDriverGetVersion",
        )?;
        let mut major = 0;
        let mut minor = 0;
        inner.api.check(
            unsafe {
                (inner.api.device_get_attribute)(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                )
            },
            "cuDeviceGetAttribute(compute_major)",
        )?;
        inner.api.check(
            unsafe {
                (inner.api.device_get_attribute)(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                )
            },
            "cuDeviceGetAttribute(compute_minor)",
        )?;
        Ok(Self {
            inner,
            driver_version,
            architecture: (major, minor),
        })
    }

    pub fn driver_version(&self) -> i32 {
        self.driver_version
    }

    pub fn architecture(&self) -> (i32, i32) {
        self.architecture
    }

    pub fn stream(&self) -> Result<CudaStream, CudaError> {
        self.inner.current()?;
        let mut stream = ptr::null_mut();
        self.inner.api.check(
            unsafe { (self.inner.api.stream_create)(&mut stream, CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        )?;
        Ok(CudaStream {
            inner: Arc::clone(&self.inner),
            raw: stream,
        })
    }

    pub fn load_module(&self, image: &[u8]) -> Result<CudaModule, CudaError> {
        self.inner.current()?;
        let mut module = ptr::null_mut();
        self.inner.api.check(
            unsafe { (self.inner.api.module_load_data)(&mut module, image.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        Ok(CudaModule {
            inner: Arc::clone(&self.inner),
            raw: module,
        })
    }

    pub fn alloc(&self, bytes: usize) -> Result<DeviceAllocation, CudaError> {
        self.inner.current()?;
        let mut raw = 0;
        // CUDA rejects a zero-byte allocation; reserve one byte so callers can
        // safely pass a null-length table without a special ownership shape.
        let allocated = bytes.max(1);
        self.inner.api.check(
            unsafe { (self.inner.api.mem_alloc)(&mut raw, allocated) },
            "cuMemAlloc",
        )?;
        Ok(DeviceAllocation {
            inner: Arc::clone(&self.inner),
            raw,
            bytes,
        })
    }
}

pub struct DeviceAllocation {
    inner: Arc<ContextInner>,
    raw: DevicePtr,
    bytes: usize,
}

unsafe impl Send for DeviceAllocation {}
unsafe impl Sync for DeviceAllocation {}

impl fmt::Debug for DeviceAllocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceAllocation")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl DeviceAllocation {
    pub fn ptr(&self) -> DevicePtr {
        self.raw
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        let _ = self.inner.current();
        unsafe { (self.inner.api.mem_free)(self.raw) };
    }
}

pub struct CudaModule {
    inner: Arc<ContextInner>,
    raw: CuModule,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    pub fn function(&self, symbol: &str) -> Result<CudaFunction, CudaError> {
        self.inner.current()?;
        let symbol = CString::new(symbol).map_err(|_| CudaError {
            call: "cuModuleGetFunction".to_owned(),
            status: -1,
            name: "CUDA_ERROR_INVALID_VALUE".to_owned(),
            detail: "kernel symbol contains a NUL byte".to_owned(),
        })?;
        let mut raw = ptr::null_mut();
        self.inner.api.check(
            unsafe { (self.inner.api.module_get_function)(&mut raw, self.raw, symbol.as_ptr()) },
            format!("cuModuleGetFunction({})", symbol.to_string_lossy()),
        )?;
        Ok(CudaFunction {
            inner: Arc::clone(&self.inner),
            raw,
        })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        let _ = self.inner.current();
        unsafe { (self.inner.api.module_unload)(self.raw) };
    }
}

#[derive(Clone)]
pub struct CudaFunction {
    inner: Arc<ContextInner>,
    raw: CuFunction,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

pub struct CudaStream {
    inner: Arc<ContextInner>,
    raw: CuStream,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn copy_to_device<T: Copy>(
        &self,
        destination: DevicePtr,
        source: &[T],
    ) -> Result<(), CudaError> {
        self.inner.current()?;
        let bytes = std::mem::size_of_val(source);
        if bytes == 0 {
            return Ok(());
        }
        self.inner.api.check(
            unsafe {
                (self.inner.api.memcpy_htod_async)(
                    destination,
                    source.as_ptr().cast(),
                    bytes,
                    self.raw,
                )
            },
            "cuMemcpyHtoDAsync",
        )
    }

    pub fn copy_from_device<T: Copy>(
        &self,
        destination: &mut [T],
        source: DevicePtr,
    ) -> Result<(), CudaError> {
        self.inner.current()?;
        let bytes = std::mem::size_of_val(destination);
        if bytes == 0 {
            return Ok(());
        }
        self.inner.api.check(
            unsafe {
                (self.inner.api.memcpy_dtoh_async)(
                    destination.as_mut_ptr().cast(),
                    source,
                    bytes,
                    self.raw,
                )
            },
            "cuMemcpyDtoHAsync",
        )
    }

    pub fn memset_u8(
        &self,
        destination: DevicePtr,
        value: u8,
        count: usize,
    ) -> Result<(), CudaError> {
        self.inner.current()?;
        self.inner.api.check(
            unsafe { (self.inner.api.memset_d8_async)(destination, value, count, self.raw) },
            "cuMemsetD8Async",
        )
    }

    pub fn memset_u32(
        &self,
        destination: DevicePtr,
        value: u32,
        count: usize,
    ) -> Result<(), CudaError> {
        self.inner.current()?;
        self.inner.api.check(
            unsafe { (self.inner.api.memset_d32_async)(destination, value, count, self.raw) },
            "cuMemsetD32Async",
        )
    }

    pub fn launch(
        &self,
        function: &CudaFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        args: &[KernelArg],
    ) -> Result<(), CudaError> {
        self.inner.current()?;
        if !Arc::ptr_eq(&self.inner, &function.inner) {
            return Err(CudaError {
                call: "cuLaunchKernel".to_owned(),
                status: -1,
                name: "CUDA_ERROR_INVALID_CONTEXT".to_owned(),
                detail: "function and stream belong to different contexts".to_owned(),
            });
        }
        let storage = args
            .iter()
            .copied()
            .map(ArgStorage::from)
            .collect::<Vec<_>>();
        let mut parameters = storage.iter().map(ArgStorage::address).collect::<Vec<_>>();
        self.inner.api.check(
            unsafe {
                (self.inner.api.launch_kernel)(
                    function.raw,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_bytes,
                    self.raw,
                    parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.inner.current()?;
        self.inner.api.check(
            unsafe { (self.inner.api.stream_synchronize)(self.raw) },
            "cuStreamSynchronize",
        )
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let _ = self.inner.current();
        unsafe { (self.inner.api.stream_destroy)(self.raw) };
    }
}
