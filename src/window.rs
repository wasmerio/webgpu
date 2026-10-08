//! Native windows for guest surfaces (the `window` feature).
//!
//! A guest that creates a surface gets a window, named after the selector it
//! passed. Windows belong to the thread that runs the platform's event loop,
//! which on macOS has to be the process's main thread, while a guest blocks
//! whichever thread runs it. So the two are separated: [`WindowHost::run`]
//! keeps the calling thread for the window system and runs the guest on
//! another one.
//!
//! The window system is only started when a guest asks for its first
//! surface. A program that never presents anything runs without one: no
//! application appears, and no display is needed.
//!
//! ```no_run
//! # fn run_guest(_: &wasmer_webgpu::WebGpuCtx) -> i32 { 0 }
//! use wasmer_webgpu::WebGpuCtx;
//! use wasmer_webgpu::window::{WindowExit, WindowHost};
//!
//! let host = WindowHost::new();
//! let ctx = WebGpuCtx::builder().surface_provider(host.surfaces()).build();
//! let control = ctx.runtime_control();
//! match host.run(move || run_guest(&ctx))? {
//!     WindowExit::Finished(code) => std::process::exit(code),
//!     // The user closed a window: stop the guest's GPU work and leave.
//!     WindowExit::Closed => {
//!         control.terminate_all();
//!         std::process::exit(0)
//!     }
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};

use anyhow::{Context, Result, anyhow};
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use crate::surface::{SurfaceProvider, SurfaceTarget, WindowTarget};

/// How the windows of a [`WindowHost`] look.
#[derive(Clone, Debug)]
pub struct WindowOptions {
    /// Title of a window whose surface was created with the empty selector.
    /// Other windows are titled with their selector.
    pub title: String,
    /// Initial size of a window's content, in logical pixels.
    pub size: (u32, u32),
    pub resizable: bool,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self {
            title: "Wasmer".to_owned(),
            size: (640, 480),
            resizable: true,
        }
    }
}

/// Why [`WindowHost::run`] returned.
#[derive(Debug)]
pub enum WindowExit<T> {
    /// The guest closure returned this.
    Finished(T),
    /// The user closed a window while the guest was still running. Its
    /// windows are hidden and its surfaces keep accepting frames, so the
    /// guest can be stopped (or left to finish) at the embedder's pace.
    Closed,
}

/// What other threads ask of the thread that owns the windows.
enum Request {
    Open {
        selector: String,
        reply: mpsc::Sender<Option<Arc<Window>>>,
    },
    Run(Box<dyn FnOnce() + Send>),
    Exit,
}

/// How a request reaches the owning thread.
enum Link {
    /// No window system yet: the owning thread waits on a plain channel.
    Waiting(mpsc::Sender<Request>),
    /// The event loop runs and takes requests as its user events.
    Running(EventLoopProxy<Request>),
    /// The event loop ended, or could not be started.
    Closed,
}

struct Shared {
    link: Mutex<Link>,
}

impl Shared {
    /// Hands `request` to the owning thread. `false` when nobody will take it.
    fn send(&self, request: Request) -> bool {
        match &*self
            .link
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            Link::Waiting(sender) => sender.send(request).is_ok(),
            Link::Running(proxy) => proxy.send_event(request).is_ok(),
            Link::Closed => false,
        }
    }

    fn set(&self, link: Link) {
        *self
            .link
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = link;
    }
}

/// Owns the windows guests present to.
pub struct WindowHost {
    options: WindowOptions,
    shared: Arc<Shared>,
    requests: mpsc::Receiver<Request>,
}

impl Default for WindowHost {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowHost {
    pub fn new() -> Self {
        Self::with_options(WindowOptions::default())
    }

    pub fn with_options(options: WindowOptions) -> Self {
        let (sender, requests) = mpsc::channel();
        Self {
            options,
            shared: Arc::new(Shared {
                link: Mutex::new(Link::Waiting(sender)),
            }),
            requests,
        }
    }

    /// The [`SurfaceProvider`] to build the [`crate::WebGpuCtx`] with.
    pub fn surfaces(&self) -> Arc<WindowSurfaces> {
        Arc::new(WindowSurfaces {
            shared: Arc::clone(&self.shared),
        })
    }

    /// Runs `guest` on a new thread and keeps this one for the windows, until
    /// the guest returns or the user closes a window.
    ///
    /// Call it on the main thread: that is the only thread some platforms
    /// allow a window system on.
    pub fn run<T: Send + 'static>(
        self,
        guest: impl FnOnce() -> T + Send + 'static,
    ) -> Result<WindowExit<T>> {
        let Self {
            options,
            shared,
            requests,
        } = self;
        let (finished, result) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("wasmer-webgpu-guest".to_owned())
            .spawn({
                let shared = Arc::clone(&shared);
                move || {
                    // Dropped when the guest returns or unwinds: either way
                    // this thread is done and the owner must stop waiting.
                    struct Done(Arc<Shared>);
                    impl Drop for Done {
                        fn drop(&mut self) {
                            self.0.send(Request::Exit);
                        }
                    }
                    let _done = Done(shared);
                    let _ = finished.send(guest());
                }
            })
            .context("failed to start the guest's thread")?;

        // Until a guest wants a window there is nothing to own: wait.
        let first = loop {
            match requests.recv() {
                Ok(Request::Exit) | Err(_) => break None,
                // Nothing here has a thread of its own yet; this one will do.
                Ok(Request::Run(work)) => work(),
                Ok(request @ Request::Open { .. }) => break Some(request),
            }
        };

        if let Some(first) = first {
            match EventLoop::<Request>::with_user_event().build() {
                Ok(event_loop) => {
                    let proxy = event_loop.create_proxy();
                    let _ = proxy.send_event(first);
                    // From here requests go to the event loop. Whatever was
                    // sent the old way in the meantime follows the first one.
                    shared.set(Link::Running(proxy.clone()));
                    for request in requests.try_iter() {
                        let _ = proxy.send_event(request);
                    }
                    let mut app = App {
                        options,
                        windows: HashMap::new(),
                        closed: false,
                    };
                    let outcome = event_loop.run_app(&mut app);
                    shared.set(Link::Closed);
                    outcome.context("the window system's event loop failed")?;
                    if app.closed {
                        // The guest may be in the middle of a frame. Its
                        // surfaces stay valid for as long as their windows
                        // exist, so the windows are hidden and kept rather
                        // than destroyed.
                        for window in app.windows.values() {
                            window.set_visible(false);
                        }
                        std::mem::forget(app.windows);
                        return Ok(WindowExit::Closed);
                    }
                }
                Err(error) => {
                    // No window system here (a machine without a display):
                    // guests get no surfaces, and run on regardless.
                    shared.set(Link::Closed);
                    eprintln!("wasmer-webgpu: cannot open windows: {error}");
                    drop(first);
                }
            }
        }

        // The guest's thread has ended, or is running without windows.
        drop(requests);
        match thread.join() {
            Ok(()) => result
                .try_recv()
                .map(WindowExit::Finished)
                .map_err(|_| anyhow!("the guest's thread ended without a result")),
            Err(_) => Err(anyhow!("the guest's thread panicked")),
        }
    }
}

struct App {
    options: WindowOptions,
    /// One window per selector: a guest that recreates its surface gets the
    /// window it had.
    windows: HashMap<String, Arc<Window>>,
    /// The loop ended because the user closed a window.
    closed: bool,
}

impl App {
    fn open(&mut self, event_loop: &ActiveEventLoop, selector: String) -> Option<Arc<Window>> {
        if let Some(window) = self.windows.get(&selector) {
            return Some(Arc::clone(window));
        }
        let title = if selector.is_empty() {
            self.options.title.clone()
        } else {
            format!("{} ({selector})", self.options.title)
        };
        let attributes = Window::default_attributes()
            .with_title(title)
            .with_inner_size(LogicalSize::new(self.options.size.0, self.options.size.1))
            .with_resizable(self.options.resizable);
        match event_loop.create_window(attributes) {
            Ok(window) => {
                let window = Arc::new(window);
                self.windows.insert(selector, Arc::clone(&window));
                Some(window)
            }
            Err(error) => {
                eprintln!("wasmer-webgpu: cannot open a window: {error}");
                None
            }
        }
    }
}

impl ApplicationHandler<Request> for App {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn user_event(&mut self, event_loop: &ActiveEventLoop, request: Request) {
        match request {
            Request::Open { selector, reply } => {
                let _ = reply.send(self.open(event_loop, selector));
            }
            Request::Run(work) => work(),
            Request::Exit => event_loop.exit(),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window: WindowId,
        event: WindowEvent,
    ) {
        // Drawing is the guest's business, from its own thread: the only
        // thing the windows themselves decide is when the user is done.
        if matches!(event, WindowEvent::CloseRequested) {
            self.closed = true;
            event_loop.exit();
        }
    }
}

/// Opens a window for each selector guests create surfaces with. Made by
/// [`WindowHost::surfaces`].
pub struct WindowSurfaces {
    shared: Arc<Shared>,
}

impl SurfaceProvider for WindowSurfaces {
    fn open(&self, selector: &str) -> Option<SurfaceTarget> {
        let (reply, answer) = mpsc::channel();
        if !self.shared.send(Request::Open {
            selector: selector.to_owned(),
            reply,
        }) {
            return None;
        }
        // A request nobody takes is dropped, and its sender with it.
        let window = answer.recv().ok()??;
        Some(SurfaceTarget::Window(Arc::new(GuestWindow {
            window,
            shared: Arc::clone(&self.shared),
        })))
    }
}

struct GuestWindow {
    window: Arc<Window>,
    shared: Arc<Shared>,
}

impl HasWindowHandle for GuestWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        self.window.window_handle()
    }
}

impl HasDisplayHandle for GuestWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.window.display_handle()
    }
}

impl WindowTarget for GuestWindow {
    fn size(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }

    fn run_on_owner_thread(&self, work: Box<dyn FnOnce() + Send>) {
        let (done, wait) = mpsc::channel();
        let sent = self.shared.send(Request::Run(Box::new(move || {
            work();
            let _ = done.send(());
        })));
        if sent {
            // Ends without the work having run when the loop has ended.
            let _ = wait.recv();
        }
    }
}
