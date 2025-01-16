use emerge_engine::{
    input::{KeyboardManager, MouseManager, mouse::MouseButton},
    window::WindowManager,
};
use winit::{
    event::{Event, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
};

fn main() {
    println!("Starting basic input example with mouse and keyboard...");

    let event_loop = EventLoop::new();
    let window_manager = WindowManager::new(&event_loop);
    let mut keyboard_manager = KeyboardManager::new();
    let mut mouse_manager = MouseManager::new();

    // Track the last cursor position
    let mut last_cursor_position = (0.0, 0.0);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        // Pass events to input managers
        keyboard_manager.handle_event(&event);
        mouse_manager.handle_event(&event);

        match event {
            Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    println!("Close requested, exiting...");
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                // Keyboard events
                if keyboard_manager.is_key_just_pressed(VirtualKeyCode::A) {
                    println!("Key 'A' just pressed!");
                }
                if keyboard_manager.is_key_held(VirtualKeyCode::A) {
                    println!("Key 'A' is being held.");
                }
                if keyboard_manager.is_key_just_released(VirtualKeyCode::A) {
                    println!("Key 'A' just released!");
                }

                // Print all keyboard events for this frame
                for event in keyboard_manager.get_events() {
                    println!(
                        "Key: {:?}, State: {:?}, Repeat: {}",
                        event.key_code, event.state, event.repeat
                    );
                }

                // Mouse events
                if mouse_manager.is_button_just_pressed(MouseButton::Left) {
                    println!("Left mouse button just pressed!");
                }
                if mouse_manager.is_button_held(MouseButton::Left) {
                    println!("Left mouse button is being held.");
                }
                if mouse_manager.is_button_just_released(MouseButton::Left) {
                    println!("Left mouse button just released!");
                }

                // Print all mouse events for this frame
                for event in mouse_manager.get_events() {
                    println!(
                        "Mouse Button: {:?}, State: {:?}, Repeat: {}",
                        event.button, event.state, event.repeat
                    );
                }

                // Accumulated motion
                let motion = mouse_manager.get_accumulated_motion();
                if motion != (0.0, 0.0) {
                    println!("Mouse motion: x = {}, y = {}", motion.0, motion.1);
                }

                // Accumulated scroll
                let scroll = mouse_manager.get_accumulated_scroll();
                if scroll != (0.0, 0.0) {
                    println!("Mouse scroll: x = {}, y = {}", scroll.0, scroll.1);
                }

                // Cursor position tracking - only print if it has changed
                let cursor_position = mouse_manager.get_cursor_position();
                if cursor_position != last_cursor_position {
                    println!("Cursor Position: x = {}, y = {}", cursor_position.0, cursor_position.1);
                    last_cursor_position = cursor_position; // Update the last position
                }

                // Exit on Escape key press
                if keyboard_manager.is_key_just_pressed(VirtualKeyCode::Escape) {
                    println!("Escape key pressed! Exiting...");
                    *control_flow = ControlFlow::Exit;
                }

                // End frame to clear one-shot states
                keyboard_manager.end_frame();
                mouse_manager.end_frame();
            }
            _ => {}
        }
    });
}
