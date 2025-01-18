pub mod keyboard;
pub mod mouse;

pub use keyboard::KeyboardManager;
pub use mouse::MouseManager;


pub fn initialize_input() -> KeyboardManager {
    println!("Input system initialized!");
    KeyboardManager::new()
}
