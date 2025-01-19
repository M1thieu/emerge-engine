//! This module is currently deprecated and used only as a playground.
//! The engine directly uses `nalgebra` for math functionality.

pub mod primitives; // Contains custom Vec2, Rect, etc.
pub mod geometry; // Contains custom math functions.

pub use primitives::*; // Re-export all custom primitives globally.
pub use geometry::*; // Re-export all custom math functions.