use crate::primitives::Vec2; // Import Vec2 from primitives
//very basic point geometry

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub position: Vec2, // Use Vec2 for the position
}

impl Point {
    /// Creates a new point.
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            position: Vec2::new(x, y),
        }
    }

    /// Calculates the distance to another point.
    pub fn distance_to(&self, other: &Point) -> f32 {
        self.position.distance(&other.position) // Pass a reference to `distance`
    }
}
