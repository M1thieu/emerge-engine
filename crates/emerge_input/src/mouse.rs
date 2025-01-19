//! Mouse input functionality for Emerge Engine.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use winit::event::{ElementState, Event, MouseButton as WinitMouseButton, MouseScrollDelta, WindowEvent};

/// Represents a mouse button event.
pub struct MouseInput {
    pub button: MouseButton,
    pub state: ElementState,
    pub repeat: bool,
}

/// Enum for mouse buttons.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Other(u16),
}

impl From<WinitMouseButton> for MouseButton {
    fn from(button: WinitMouseButton) -> Self {
        match button {
            WinitMouseButton::Left => MouseButton::Left,
            WinitMouseButton::Right => MouseButton::Right,
            WinitMouseButton::Middle => MouseButton::Middle,
            WinitMouseButton::Other(id) => MouseButton::Other(id),
        }
    }
}

/// Represents a mouse wheel event.
pub struct MouseWheelInput {
    pub delta: (f32, f32), // (x, y) scroll values
}

/// Tracks the state of the mouse.
pub struct MouseManager {
    buttons_pressed: HashMap<MouseButton, Instant>,
    buttons_held: HashMap<MouseButton, Instant>,
    buttons_released: HashMap<MouseButton, Instant>,
    wheel_events: Vec<MouseWheelInput>,
    events: Vec<MouseInput>,
    debounce_duration: Duration,
    cursor_position: (f32, f32), // Track cursor position
    delta_motion: (f32, f32),    // Accumulated motion
    delta_scroll: (f32, f32),    // Accumulated scroll
}

impl MouseManager {
    pub fn new() -> Self {
        MouseManager {
            buttons_pressed: HashMap::new(),
            buttons_held: HashMap::new(),
            buttons_released: HashMap::new(),
            wheel_events: Vec::new(),
            events: Vec::new(),
            debounce_duration: Duration::from_millis(100), // Adjust as needed
            cursor_position: (0.0, 0.0), // Default to origin
            delta_motion: (0.0, 0.0),   // Default to no motion
            delta_scroll: (0.0, 0.0),   // Default to no scroll
        }
    }

    /// Updates the cursor position and accumulates motion.
    pub fn update_cursor_position(&mut self, position: (f32, f32)) {
        self.delta_motion.0 += position.0 - self.cursor_position.0;
        self.delta_motion.1 += position.1 - self.cursor_position.1;
        self.cursor_position = position;
    }

    /// Retrieves the current cursor position.
    pub fn get_cursor_position(&self) -> (f32, f32) {
        self.cursor_position
    }

    /// Retrieves the accumulated mouse motion for the current frame.
    pub fn get_accumulated_motion(&self) -> (f32, f32) {
        self.delta_motion
    }

    /// Retrieves the accumulated scroll values for the current frame.
    pub fn get_accumulated_scroll(&self) -> (f32, f32) {
        self.delta_scroll
    }

    /// Handles window events to update mouse button, wheel, and cursor states.
    pub fn handle_event(&mut self, event: &Event<()>) {
        if let Event::WindowEvent { event, .. } = event {
            match event {
                // Handle cursor movement and accumulate motion
                WindowEvent::CursorMoved { position, .. } => {
                    self.update_cursor_position((position.x as f32, position.y as f32));
                }
                // Handle mouse button input
                WindowEvent::MouseInput { state, button, .. } => {
                    let now = Instant::now();
                    let mouse_button: MouseButton = (*button).into();

                    match state {
                        ElementState::Pressed => {
                            let is_repeat = self.buttons_held.contains_key(&mouse_button);
                            if !is_repeat {
                                self.buttons_pressed.insert(mouse_button, now);
                            }
                            self.buttons_held.insert(mouse_button, now);
                            self.events.push(MouseInput {
                                button: mouse_button,
                                state: ElementState::Pressed,
                                repeat: is_repeat,
                            });
                        }
                        ElementState::Released => {
                            self.buttons_released.insert(mouse_button, now);
                            self.buttons_held.remove(&mouse_button);
                            self.events.push(MouseInput {
                                button: mouse_button,
                                state: ElementState::Released,
                                repeat: false,
                            });
                        }
                    }
                }
                // Handle mouse wheel input and accumulate scroll
                WindowEvent::MouseWheel { delta, .. } => {
                    let (x, y) = match delta {
                        MouseScrollDelta::LineDelta(dx, dy) => (*dx, *dy),
                        MouseScrollDelta::PixelDelta(pos) => (pos.x as f32, pos.y as f32),
                    };
                    self.delta_scroll.0 += x;
                    self.delta_scroll.1 += y;
                    self.wheel_events.push(MouseWheelInput { delta: (x, y) });
                }
                _ => {}
            }
        }
    }

    /// Clears transient states (pressed/released events, wheel events, etc.) and resets accumulations.
    pub fn end_frame(&mut self) {
        self.buttons_pressed.clear();
        self.buttons_released.clear();
        self.events.clear();
        self.wheel_events.clear();
        self.delta_motion = (0.0, 0.0); // Reset accumulated motion
        self.delta_scroll = (0.0, 0.0); // Reset accumulated scroll
    }

    /// Checks if a mouse button was just pressed.
    pub fn is_button_just_pressed(&self, button: MouseButton) -> bool {
        self.buttons_pressed.contains_key(&button)
    }

    /// Checks if a mouse button is being held.
    pub fn is_button_held(&self, button: MouseButton) -> bool {
        if let Some(&last_held) = self.buttons_held.get(&button) {
            Instant::now().duration_since(last_held) >= self.debounce_duration
        } else {
            false
        }
    }

    /// Checks if a mouse button was just released.
    pub fn is_button_just_released(&self, button: MouseButton) -> bool {
        self.buttons_released.contains_key(&button)
    }

    /// Retrieves all mouse button events for the current frame.
    pub fn get_events(&self) -> &[MouseInput] {
        &self.events
    }

    /// Retrieves all mouse wheel events for the current frame.
    pub fn get_wheel_events(&self) -> &[MouseWheelInput] {
        &self.wheel_events
    }
}
