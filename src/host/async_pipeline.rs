//! `wgpuDeviceCreate{Compute,Render}PipelineAsync` on a compiler lane.
//!
//! The pipeline's objects are resolved into an owned job on the guest's
//! thread; no guest address, store or Host is sent. The lane compiles it and
//! keeps the outcome until the guest next polls, which publishes the handle
//! and the future's event. When the lane is at capacity, the guest's thread
//! compiles the job itself, as every async creation did before.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::pipeline_worker::{Rejected, Slot};
use super::{DeviceState, Host, InstanceState, Object, ObjectKind, Record, lock};
use crate::abi::enums::{CreatePipelineAsyncStatus, ErrorType};
use crate::abi::structs::{ComputePipelineDescriptor, RenderPipelineDescriptor};
use crate::abi::{AbiEnum, EVENT_CREATE_COMPUTE_PIPELINE, EVENT_CREATE_RENDER_PIPELINE, Handle};

pub(super) type Outcome = Result<Object, (ErrorType, String)>;

/// Who a job's outcome is for, checked again when it is published.
pub(super) struct Request {
    instance: Handle,
    instance_state: Arc<InstanceState>,
    device: Arc<DeviceState>,
    future: u64,
    kind: u32,
}

impl Host {
    fn pipeline_request(&self, device: Handle, kind: u32, future: u64) -> Result<Request, String> {
        let device_state = self.device(device).ok_or("invalid device.")?;
        let instance = self.instance_of(device);
        let instance_state = self
            .instance(instance)
            .ok_or("the device's instance was released.")?;
        Ok(Request {
            instance,
            instance_state,
            device: device_state,
            kind,
            future,
        })
    }

    /// Hands `create` to the compiler lane, or runs it here when the lane is
    /// at capacity. Its errors are captured as the future's outcome, never
    /// raised on the device.
    fn enqueue_pipeline(
        &self,
        request: Request,
        function: &'static str,
        create: impl FnOnce() -> Result<Object, String> + Send + 'static,
    ) {
        let device = Arc::clone(&request.device);
        let instance = Arc::clone(&request.instance_state);
        let job = move || -> Outcome {
            // Its result would be retired unpublished: don't compile it.
            if !instance.active.load(Ordering::Acquire) {
                return Err((
                    ErrorType::Internal,
                    "The pipeline's instance was released.".into(),
                ));
            }
            device.catching(create).and_then(|created| {
                created.map_err(|message| (ErrorType::Validation, format!("{function}: {message}")))
            })
        };
        match self.pipelines.enqueue(request.instance, request, job) {
            Ok(()) => {}
            Err(Rejected::Busy(request, job)) => {
                let outcome = job();
                self.publish_pipeline(request, Ok(outcome), None);
            }
            Err(Rejected::Stopped(request)) => self.publish_pipeline(
                request,
                Err("The pipeline compiler was stopped.".into()),
                None,
            ),
        }
    }

    pub(crate) fn enqueue_compute_pipeline(
        &self,
        device: Handle,
        descriptor: ComputePipelineDescriptor,
        future: u64,
    ) {
        const FUNCTION: &str = "wgpuDeviceCreateComputePipelineAsync";
        let prepared = self
            .pipeline_request(device, EVENT_CREATE_COMPUTE_PIPELINE, future)
            .and_then(|request| {
                let job = self.compute_job(Arc::clone(&request.device), descriptor)?;
                Ok((request, job))
            });
        match prepared {
            Ok((request, job)) => self.enqueue_pipeline(request, FUNCTION, move || job.create()),
            Err(message) => self.fail_pipeline_async(
                EVENT_CREATE_COMPUTE_PIPELINE,
                future,
                format!("{FUNCTION}: {message}"),
            ),
        }
    }

    pub(crate) fn enqueue_render_pipeline(
        &self,
        device: Handle,
        descriptor: RenderPipelineDescriptor,
        future: u64,
    ) {
        const FUNCTION: &str = "wgpuDeviceCreateRenderPipelineAsync";
        let prepared = self
            .pipeline_request(device, EVENT_CREATE_RENDER_PIPELINE, future)
            .and_then(|request| {
                let job = self.render_job(Arc::clone(&request.device), descriptor)?;
                Ok((request, job))
            });
        match prepared {
            Ok((request, job)) => self.enqueue_pipeline(request, FUNCTION, move || job.create()),
            Err(message) => self.fail_pipeline_async(
                EVENT_CREATE_RENDER_PIPELINE,
                future,
                format!("{FUNCTION}: {message}"),
            ),
        }
    }

    /// Publishes what the compiler lane finished since the last call.
    pub(crate) fn publish_pipeline_completions(&self) {
        for completed in self.pipelines.take_completed() {
            self.publish_pipeline(completed.info, completed.result, Some(completed.slot));
        }
    }

    /// Registers a compiled pipeline and queues its future's event. `Err` is a
    /// lane failure (a panic or a stopped lane); `slot` is the lane admission
    /// the event keeps until the guest collects it.
    fn publish_pipeline(
        &self,
        request: Request,
        result: Result<Outcome, String>,
        slot: Option<Arc<Slot>>,
    ) {
        let result = result.unwrap_or_else(|message| Err((ErrorType::Internal, message)));
        let mut object = None;
        let mut error = None;
        match result {
            Ok(value) => object = Some(value),
            Err(value) => error = Some(value),
        }
        let mut table = lock(&self.table);
        let live_instance = matches!(table.get(request.instance), Some(Object::Instance(state))
            if Arc::ptr_eq(&state, &request.instance_state));
        if !live_instance {
            // Instance release already supplied guest cancellation. Retire
            // native results without inserting a handle or another callback.
            drop(table);
            return;
        }
        // A lost device doesn't fail the creation. As in WebGPU (and Emdawn),
        // the pipeline is made, unusable, and the loss is reported through the
        // device's own future.
        if self.is_terminated() {
            error = Some((ErrorType::Internal, "The host was terminated.".into()));
        }
        let mut handle = 0;
        if error.is_none() {
            handle = table.insert(
                object.take().expect("successful compiler result"),
                request.instance,
            );
            if handle == 0 {
                error = Some((ErrorType::Internal, "Pipeline object limit reached.".into()));
            }
        }
        let (status, message) = match error {
            None => (CreatePipelineAsyncStatus::Success, String::new()),
            Some((ErrorType::Validation, message)) => {
                (CreatePipelineAsyncStatus::ValidationError, message)
            }
            Some((_, message)) => (CreatePipelineAsyncStatus::InternalError, message),
        };
        // The table lock serializes instance release with publication; the
        // embedder hears of the event once it is released.
        let queued = self.events.queue(Record {
            future: request.future,
            kind: request.kind,
            status: status.as_u32(),
            handle,
            message,
            _keep_alive: slot.map(|slot| slot as Arc<dyn std::any::Any + Send + Sync>),
            pipeline_instance: request.instance,
            ..Record::default()
        });
        drop(table);
        if queued {
            self.events.wake_owner();
        }
        drop(object);
    }

    pub(crate) fn discard_pipeline_instance(&self, instance: Handle) {
        self.pipelines.discard(instance);
        for record in self.events.retire_pipeline_instance(instance) {
            let kind = match record.kind {
                EVENT_CREATE_RENDER_PIPELINE => ObjectKind::RenderPipeline,
                EVENT_CREATE_COMPUTE_PIPELINE => ObjectKind::ComputePipeline,
                _ => continue,
            };
            if record.handle != 0 {
                self.release(record.handle, kind);
            }
        }
    }
}

#[cfg(test)]
mod native_gpu_tests {
    // Run explicitly with --ignored --test-threads=1 in an exclusive GPU slot.
    // The held job occupies the SAME production compiler lane, so descriptor
    // retirement/cancellation precedes every actual driver call deterministically.
    use super::super::bind::PipelineLayoutState;
    use super::super::shader::ShaderState;
    use super::*;
    use crate::WebGpuCtx;
    use crate::abi::enums::TextureFormat;
    use crate::abi::enums::{ErrorFilter, PrimitiveTopology, color_write_mask};
    use crate::abi::structs::ConstantEntry;
    use crate::abi::structs::{
        ColorTargetState, ComputeState, FragmentState, MultisampleState, PrimitiveState,
        VertexState,
    };
    use crate::host::scopes;
    use std::borrow::Cow;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    const HOLD_FUTURE: u64 = 9001;
    const COMPUTE: &str = "override scale: u32 = 1u;\n@group(0) @binding(0) var<storage,read_write> data: array<u32>;\n@compute @workgroup_size(1) fn main() { data[0] *= scale; }";
    const RENDER: &str = "@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { return vec4f(f32(i),0.0,0.0,1.0); }\n@fragment fn fs()->@location(0) vec4f { return vec4f(1.0); }";

    fn fixture() -> (Box<Host>, Handle, Handle) {
        fixture_with(Box::new(Host::new(Arc::clone(&WebGpuCtx::default().inner))))
    }

    fn fixture_with<H: std::ops::Deref<Target = Host>>(host: H) -> (H, Handle, Handle) {
        let native = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(native.request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: std::env::var_os("WEBGPU_TEST_FALLBACK_ADAPTER").is_some(),
            ..Default::default()
        }))
        .expect("native GPU adapter is required, never silently skip");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("async-owned-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("native GPU device");
        let instance = host.insert(
            Object::Instance(Arc::new(InstanceState {
                instance: native,
                active: AtomicBool::new(true),
            })),
            0,
        );
        let state = DeviceState::new(&host, device, queue, adapter.get_info(), 9000);
        let device = host.insert(Object::Device(Arc::clone(&state)), instance);
        state.link.handle.store(device, Ordering::Release);
        host.register_device(&state);
        (host, instance, device)
    }

    fn shader(host: &Host, device: Handle, source: &str) -> Handle {
        let state = host.device(device).unwrap();
        let module = state
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("owned-test-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            });
        host.insert(
            Object::ShaderModule(Arc::new(ShaderState {
                module,
                device: state,
            })),
            host.instance_of(device),
        )
    }

    fn explicit_layout(host: &Host, device: Handle, storage: bool) -> (Handle, Handle) {
        let state = host.device(device).unwrap();
        let entries = [wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        let bgl = state
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("owned-test-bgl"),
                entries: if storage { &entries } else { &[] },
            });
        let layout = state
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("owned-test-layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        (
            host.insert(
                Object::PipelineLayout(Arc::new(PipelineLayoutState { layout })),
                host.instance_of(device),
            ),
            host.insert(
                Object::BindGroupLayout(Arc::new(bgl)),
                host.instance_of(device),
            ),
        )
    }

    fn hold(host: &Host, device: Handle) -> mpsc::Sender<()> {
        let request = host
            .pipeline_request(device, EVENT_CREATE_COMPUTE_PIPELINE, HOLD_FUTURE)
            .unwrap();
        let (entered, entry) = mpsc::channel();
        let (release, released) = mpsc::channel();
        host.pipelines
            .enqueue(request.instance, request, move || {
                entered.send(()).unwrap();
                released
                    .recv_timeout(Duration::from_secs(15))
                    .expect("test releases held lane");
                Err((ErrorType::Internal, "test-only held job".into()))
            })
            .unwrap_or_else(|_| panic!("held-job admission"));
        entry
            .recv_timeout(Duration::from_secs(5))
            .expect("same worker is held");
        release
    }

    fn compute_descriptor(
        module: Handle,
        layout: Handle,
        entry: &str,
    ) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: "owned-compute-label".into(),
            layout,
            compute: ComputeState {
                module,
                entry_point: Some(entry.into()),
                constants: vec![ConstantEntry {
                    key: "scale".into(),
                    value: 3.0,
                    ..Default::default()
                }],
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn render_descriptor(module: Handle, layout: Handle) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: "owned-render-label".into(),
            layout,
            vertex: VertexState {
                module,
                entry_point: Some("vs".into()),
                ..Default::default()
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            multisample: MultisampleState {
                count: 1,
                mask: u32::MAX,
                ..Default::default()
            },
            fragment: Some(Box::new(FragmentState {
                module,
                entry_point: Some("fs".into()),
                targets: vec![ColorTargetState {
                    format: TextureFormat::RGBA8Unorm,
                    write_mask: color_write_mask::ALL,
                    ..Default::default()
                }],
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    thread_local! {
        /// Records a `wait` collected for other futures, kept for the waits
        /// and drains after it.
        static UNCLAIMED: std::cell::RefCell<Vec<Record>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Publishes what the lane has finished, and returns every record not
    /// claimed yet.
    fn drain(host: &Host) -> Vec<Record> {
        host.publish_pipeline_completions();
        let mut records = UNCLAIMED.with_borrow_mut(std::mem::take);
        records.extend(host.events.take_for_test());
        assert!(
            !records
                .iter()
                .any(|r| r.kind == crate::abi::EVENT_UNCAPTURED_ERROR),
            "job failure leaked into device errors"
        );
        records
    }

    /// The record of `future`, once it is published. Other records are kept.
    fn wait(host: &Host, future: u64) -> Record {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let epoch = host.events.epoch();
            let (mut matches, others): (Vec<_>, Vec<_>) =
                drain(host).into_iter().partition(|r| r.future == future);
            UNCLAIMED.with_borrow_mut(|unclaimed| unclaimed.extend(others));
            assert!(matches.len() <= 1, "exact-once native outcome");
            if let Some(record) = matches.pop() {
                return record;
            }
            assert!(
                Instant::now() < deadline,
                "native pipeline completion timed out"
            );
            host.events.wait_for_push(Duration::from_millis(50), epoch);
        }
    }

    /// Collects the outcome of the job `hold` ran.
    fn retire_hold(host: &Host) {
        let record = wait(host, HOLD_FUTURE);
        assert_eq!(
            record.status,
            CreatePipelineAsyncStatus::InternalError.as_u32()
        );
    }

    fn assert_only_the_loss_is_left(host: &Host) {
        let kinds: Vec<_> = drain(host).iter().map(|r| r.kind).collect();
        assert_eq!(kinds, [crate::abi::EVENT_DEVICE_LOST]);
    }

    fn assert_published_pipeline(record: &Record) {
        assert_eq!(
            (record.status, record.message.as_str()),
            (CreatePipelineAsyncStatus::Success.as_u32(), "")
        );
        assert_ne!(record.handle, 0);
    }

    #[test]
    #[ignore = "requires native GPU and exclusive serial test slot"]
    fn native_compute_and_render_survive_released_inputs() {
        let (host, _instance, device) = fixture();
        let release = hold(&host, device);
        let compute_module = shader(&host, device, COMPUTE);
        let render_module = shader(&host, device, RENDER);
        let (compute_layout, compute_bgl) = explicit_layout(&host, device, true);
        let (render_layout, render_bgl) = explicit_layout(&host, device, false);
        let compute_weak = Arc::downgrade(&host.shader_module(compute_module).unwrap());
        let layout_weak = Arc::downgrade(&host.pipeline_layout(compute_layout).unwrap());
        host.enqueue_compute_pipeline(
            device,
            compute_descriptor(compute_module, compute_layout, "main"),
            1,
        );
        host.enqueue_render_pipeline(device, render_descriptor(render_module, render_layout), 2);
        for module in [compute_module, render_module] {
            host.release(module, ObjectKind::ShaderModule);
        }
        for layout in [compute_layout, render_layout] {
            host.release(layout, ObjectKind::PipelineLayout);
        }
        for bgl in [compute_bgl, render_bgl] {
            host.release(bgl, ObjectKind::BindGroupLayout);
        }
        assert!(
            compute_weak.upgrade().is_some() && layout_weak.upgrade().is_some(),
            "queued ownership"
        );
        assert_eq!(
            host.pipelines.completed_count_for_test(),
            0,
            "no inline compilation"
        );
        release.send(()).unwrap();
        // Drain once both jobs finish, so the first wait cannot retire the other
        // event before its assertion. Native event order is not an API promise.
        let deadline = Instant::now() + Duration::from_secs(15);
        while host.pipelines.completed_count_for_test() < 3 {
            let epoch = host.events.epoch();
            assert!(Instant::now() < deadline);
            host.events.wait_for_push(Duration::from_millis(50), epoch);
        }
        retire_hold(&host);
        let records = drain(&host);
        for (future, kind) in [
            (1, ObjectKind::ComputePipeline),
            (2, ObjectKind::RenderPipeline),
        ] {
            let matches: Vec<_> = records.iter().filter(|r| r.future == future).collect();
            assert_eq!(matches.len(), 1);
            assert_eq!(
                matches[0].status,
                CreatePipelineAsyncStatus::Success.as_u32()
            );
            assert_ne!(matches[0].handle, 0);
            host.release(matches[0].handle, kind);
        }
        assert!(
            compute_weak.upgrade().is_none() && layout_weak.upgrade().is_none(),
            "inputs retired after compilation"
        );
        assert!(drain(&host).is_empty());
    }

    #[test]
    #[ignore = "requires native GPU and exclusive serial test slot"]
    fn the_event_waker_runs_outside_the_table_lock() {
        let host_slot = Arc::new(std::sync::OnceLock::<std::sync::Weak<Host>>::new());
        let (wakes, locked) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let ctx = WebGpuCtx::builder()
            .event_waker(Arc::new({
                let (host_slot, wakes, locked) = (
                    Arc::clone(&host_slot),
                    Arc::clone(&wakes),
                    Arc::clone(&locked),
                );
                move || {
                    if let Some(host) = host_slot.get().and_then(std::sync::Weak::upgrade) {
                        wakes.fetch_add(1, Ordering::SeqCst);
                        if host.table.try_lock().is_err() {
                            locked.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            }))
            .build();
        let (host, _instance, device) = fixture_with(Arc::new(Host::new(Arc::clone(&ctx.inner))));
        host_slot.set(Arc::downgrade(&host)).unwrap();
        let module = shader(&host, device, COMPUTE);
        // A result of the compiler thread, published here...
        host.enqueue_compute_pipeline(device, compute_descriptor(module, 0, "main"), 1);
        assert_eq!(
            wait(&host, 1).status,
            CreatePipelineAsyncStatus::Success.as_u32()
        );
        // ...and a creation the full lane hands back to this thread.
        let release = hold(&host, device);
        let queued = super::super::PIPELINE_LANE_CAPACITY as u64 - 1;
        for future in 2..2 + queued {
            host.enqueue_compute_pipeline(device, compute_descriptor(module, 0, "main"), future);
        }
        let wakes_before = wakes.load(Ordering::SeqCst);
        host.enqueue_compute_pipeline(device, compute_descriptor(module, 0, "main"), 2 + queued);
        let records = drain(&host);
        assert_eq!(records.len(), 1, "only the inline creation finished");
        assert!(
            wakes.load(Ordering::SeqCst) > wakes_before,
            "the inline creation woke"
        );
        release.send(()).unwrap();
        assert_eq!(
            locked.load(Ordering::SeqCst),
            0,
            "wakes inside the table lock"
        );
    }

    #[test]
    #[ignore = "requires native GPU and exclusive serial test slot"]
    fn native_full_lane_compiles_on_the_callers_thread() {
        let (host, _instance, device) = fixture();
        let release = hold(&host, device);
        let module = shader(&host, device, COMPUTE);
        // With the held job, these fill the lane.
        let queued = super::super::PIPELINE_LANE_CAPACITY as u64 - 1;
        for future in 1..=queued {
            host.enqueue_compute_pipeline(device, compute_descriptor(module, 0, "main"), future);
        }
        // The next one is compiled right here, while the lane is still held.
        let inline = queued + 1;
        host.enqueue_compute_pipeline(device, compute_descriptor(module, 0, "main"), inline);
        let records = drain(&host);
        assert_eq!(records.len(), 1, "only the inline creation finished");
        assert_eq!(records[0].future, inline);
        assert_eq!(
            records[0].status,
            CreatePipelineAsyncStatus::Success.as_u32()
        );
        assert_ne!(records[0].handle, 0);
        host.release(records[0].handle, ObjectKind::ComputePipeline);
        release.send(()).unwrap();
        retire_hold(&host);
        for future in 1..=queued {
            let record = wait(&host, future);
            assert_eq!(record.status, CreatePipelineAsyncStatus::Success.as_u32());
            host.release(record.handle, ObjectKind::ComputePipeline);
        }
        assert!(drain(&host).is_empty());
    }

    #[test]
    #[ignore = "requires native GPU and exclusive serial test slot"]
    fn native_validation_failure_is_job_local_and_exactly_once() {
        let (host, _instance, device) = fixture();
        let release = hold(&host, device);
        let module = shader(&host, device, COMPUTE);
        let id = host.device(device).unwrap().id();
        for filter in [
            ErrorFilter::OutOfMemory,
            ErrorFilter::Internal,
            ErrorFilter::Validation,
        ] {
            scopes::push(id, filter);
        }
        host.enqueue_compute_pipeline(device, compute_descriptor(module, 0, "missing_entry"), 1);
        host.release(module, ObjectKind::ShaderModule);
        release.send(()).unwrap();
        retire_hold(&host);
        let record = wait(&host, 1);
        assert_eq!(
            record.status,
            CreatePipelineAsyncStatus::ValidationError.as_u32()
        );
        assert_eq!(record.handle, 0);
        assert!(!record.message.is_empty());
        for _ in 0..3 {
            assert_eq!(scopes::pop(id), Some(None));
        }
        assert!(drain(&host).is_empty(), "no duplicate outcome");
    }

    // A lost device doesn't fail a pipeline creation, whenever the loss comes:
    // the pipeline is published, unusable, as in WebGPU.

    #[test]
    #[ignore = "requires native GPU and exclusive serial test slot"]
    fn a_device_lost_before_the_call_still_gets_its_pipeline() {
        let (host, _instance, device) = fixture();
        let module = shader(&host, device, RENDER);
        host.device(device).unwrap().destroy();
        host.enqueue_render_pipeline(device, render_descriptor(module, 0), 1);
        assert_published_pipeline(&wait(&host, 1));
        assert_only_the_loss_is_left(&host);
    }

    #[test]
    #[ignore = "requires native GPU and exclusive serial test slot"]
    fn a_device_lost_while_compiling_still_gets_its_pipeline() {
        let (host, _instance, device) = fixture();
        let release = hold(&host, device);
        let module = shader(&host, device, RENDER);
        host.enqueue_render_pipeline(device, render_descriptor(module, 0), 1);
        host.device(device).unwrap().destroy();
        release.send(()).unwrap();
        retire_hold(&host);
        assert_published_pipeline(&wait(&host, 1));
        assert_only_the_loss_is_left(&host);
    }

    #[test]
    #[ignore = "requires native GPU and exclusive serial test slot"]
    fn a_device_lost_before_publication_still_gets_its_pipeline() {
        let (host, _instance, device) = fixture();
        let release = hold(&host, device);
        let module = shader(&host, device, RENDER);
        host.enqueue_render_pipeline(device, render_descriptor(module, 0), 1);
        host.release(module, ObjectKind::ShaderModule);
        release.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        while host.pipelines.completed_count_for_test() < 2 {
            let epoch = host.events.epoch();
            assert!(Instant::now() < deadline);
            host.events.wait_for_push(Duration::from_millis(50), epoch);
        }
        assert!(
            host.pipelines
                .any_completed_for_test(|request, result| request.future == 1
                    && matches!(result, Ok(Ok(Object::RenderPipeline(_))))),
            "compiled before the loss"
        );
        host.device(device).unwrap().destroy();
        retire_hold(&host);
        assert_published_pipeline(&wait(&host, 1));
        assert_only_the_loss_is_left(&host);
    }

    #[test]
    #[ignore = "requires native GPU and exclusive serial test slot"]
    fn native_instance_discard_releases_queued_real_descriptor() {
        let (host, instance, device) = fixture();
        let release = hold(&host, device);
        let module = shader(&host, device, COMPUTE);
        let weak = Arc::downgrade(&host.shader_module(module).unwrap());
        host.enqueue_compute_pipeline(device, compute_descriptor(module, 0, "main"), 1);
        host.release(module, ObjectKind::ShaderModule);
        let Some(Object::Instance(state)) = host.release(instance, ObjectKind::Instance) else {
            panic!("last instance");
        };
        // Same native half of instance release; actual guest CallbackCancelled
        // behavior belongs to the C shim integration test, not this unit test.
        state.active.store(false, Ordering::Release);
        host.discard_pipeline_instance(instance);
        assert!(
            weak.upgrade().is_none(),
            "queued shader ownership retired immediately"
        );
        release.send(()).unwrap();
        // Neither the held job nor this one is published for a released instance.
        let deadline = Instant::now() + Duration::from_secs(15);
        while host.pipelines.is_working() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(drain(&host).is_empty());
    }

    #[test]
    #[ignore = "requires native GPU and exclusive serial test slot"]
    fn native_termination_returns_error_for_queued_real_pipeline() {
        let (host, _instance, device) = fixture();
        let release = hold(&host, device);
        let module = shader(&host, device, COMPUTE);
        host.enqueue_compute_pipeline(device, compute_descriptor(module, 0, "main"), 1);
        host.release(module, ObjectKind::ShaderModule);
        host.terminate();
        let record = wait(&host, 1);
        assert_eq!(
            record.status,
            CreatePipelineAsyncStatus::InternalError.as_u32()
        );
        assert_eq!(record.handle, 0);
        release.send(()).unwrap();
        assert!(!drain(&host).iter().any(|r| r.future == 1));
    }
}
