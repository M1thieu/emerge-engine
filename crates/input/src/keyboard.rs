use std::collections::HashMap;
use std::time::{Duration, Instant};
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};

// Represents a key input event
pub struct KeyboardInput {
    pub key_code: VirtualKeyCode,
    pub state: ElementState,
    pub repeat: bool,
    pub text: Option<String>,
}

// Manages keyboard states
pub struct KeyboardManager {
    keys_pressed: HashMap<VirtualKeyCode, Instant>,
    keys_held: HashMap<VirtualKeyCode, Instant>,
    keys_released: HashMap<VirtualKeyCode, Instant>,
    events: Vec<KeyboardInput>,
    debounce_duration: Duration,
}

impl KeyboardManager {
    pub fn new() -> Self {
        KeyboardManager {
            keys_pressed: HashMap::new(),
            keys_held: HashMap::new(),
            keys_released: HashMap::new(),
            events: Vec::new(),
            debounce_duration: Duration::from_millis(100), // Adjust as needed
        }
    }

    /// Process window events to update key states
    pub fn handle_event(&mut self, event: &Event<()>) {
        if let Event::WindowEvent { event, .. } = event {
            if let WindowEvent::KeyboardInput { input, .. } = event {
                if let Some(key_code) = input.virtual_keycode {
                    let now = Instant::now();
                    match input.state {
                        ElementState::Pressed => {
                            // Set `repeat` only if the key is already in the `keys_held` map
                            let is_repeat = self.keys_held.contains_key(&key_code);
                            if !is_repeat {
                                self.keys_pressed.insert(key_code, now);
                            }
                            self.keys_held.insert(key_code, now);
                            self.events.push(KeyboardInput {
                                key_code,
                                state: ElementState::Pressed,
                                repeat: is_repeat,
                                text: None,
                            });
                        }
                        ElementState::Released => {
                            self.keys_released.insert(key_code, now);
                            self.keys_held.remove(&key_code);
                            self.events.push(KeyboardInput {
                                key_code,
                                state: ElementState::Released,
                                repeat: false,
                                text: None,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Called once per frame to clear one-shot states
    pub fn end_frame(&mut self) {
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.events.clear();
    }

    /// Check if a key was just pressed this frame
    pub fn is_key_just_pressed(&self, key: VirtualKeyCode) -> bool {
        self.keys_pressed.contains_key(&key)
    }

    /// Check if a key is being held (debounced)
    pub fn is_key_held(&self, key: VirtualKeyCode) -> bool {
        if let Some(&last_held) = self.keys_held.get(&key) {
            Instant::now().duration_since(last_held) >= self.debounce_duration
        } else {
            false
        }
    }

    /// Check if a key was just released this frame
    pub fn is_key_just_released(&self, key: VirtualKeyCode) -> bool {
        self.keys_released.contains_key(&key)
    }

    /// Retrieve all input events for this frame
    pub fn get_events(&self) -> &[KeyboardInput] {
        &self.events
    }
}
