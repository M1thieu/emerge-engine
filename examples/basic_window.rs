use emerge_engine::window::WindowManager;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    println!("Starting basic window example...");

    let event_loop = EventLoop::new();
    let mut window_manager = WindowManager::new(&event_loop);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    println!("Close requested, exiting...");
                    *control_flow = ControlFlow::Exit;
                }
                winit::event::WindowEvent::Resized(new_size) => {
                    println!("Window resized to: {:?}", new_size);
                    window_manager.update_size();
                }
                _ => {}
            },
            _ => {}
        }
    });
}
