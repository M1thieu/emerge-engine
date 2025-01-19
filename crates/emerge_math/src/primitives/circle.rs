use super::Vec2;

/// A circle primitive with a position and a radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f32,
}

impl Circle {
    /// Creates a new circle with the given center and radius.
    pub fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Checks if a point is inside the circle.
    pub fn contains(&self, point: &Vec2) -> bool {
        self.center.distance(point) <= self.radius
    }

    /// Checks if two circles intersect.
    pub fn intersects(&self, other: &Circle) -> bool {
        self.center.distance(&other.center) <= (self.radius + other.radius)
    }
}
