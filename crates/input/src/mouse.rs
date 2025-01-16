//! Mouse input functionality for Emerge Engine.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use winit::event::{ElementState, Event, MouseButton as WinitMouseButton, WindowEvent};

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

/// Tracks the state of the mouse.
pub struct MouseManager {
    buttons_pressed: HashMap<MouseButton, Instant>,
    buttons_held: HashMap<MouseButton, Instant>,
    buttons_released: HashMap<MouseButton, Instant>,
    events: Vec<MouseInput>,
    debounce_duration: Duration,
}

impl MouseManager {
    pub fn new() -> Self {
        MouseManager {
            buttons_pressed: HashMap::new(),
            buttons_held: HashMap::new(),
            buttons_released: HashMap::new(),
            events: Vec::new(),
            debounce_duration: Duration::from_millis(100), // Adjust as needed
        }
    }

    /// Handles window events to update mouse button states.
    pub fn handle_event(&mut self, event: &Event<()>) {
        if let Event::WindowEvent { event, .. } = event {
            if let WindowEvent::MouseInput { state, button, .. } = event {
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
        }
    }

    /// Clears transient states (pressed/released events) at the end of a frame.
    pub fn end_frame(&mut self) {
        self.buttons_pressed.clear();
        self.buttons_released.clear();
        self.events.clear();
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

    /// Retrieves all mouse events for the current frame.
    pub fn get_events(&self) -> &[MouseInput] {
        &self.events
    }
}
