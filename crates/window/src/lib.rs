use winit::{
    dpi::PhysicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

pub struct WindowManager {
    pub window: Window,
    pub size: PhysicalSize<u32>,
}

impl WindowManager {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let window = WindowBuilder::new()
            .with_title("Emerge Engine Window")
            .with_inner_size(PhysicalSize::new(800, 600))
            .build(event_loop)
            .expect("Failed to create window");

        let size = window.inner_size();

        WindowManager { window, size }
    }

    pub fn update_size(&mut self) {
        self.size = self.window.inner_size();
    }

    pub fn get_size(&self) -> (u32, u32) {
        (self.size.width, self.size.height)
    }
}
