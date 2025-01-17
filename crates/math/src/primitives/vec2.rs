use std::ops::{Add, Sub, Mul, Div};

/// A simple 2D vector structure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    /// Creates a new 2D vector.
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Returns the zero vector (0, 0).
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Returns the unit vector along the X-axis (1, 0).
    pub const X: Self = Self { x: 1.0, y: 0.0 };

    /// Returns the unit vector along the Y-axis (0, 1).
    pub const Y: Self = Self { x: 0.0, y: 1.0 };

    /// Calculates the magnitude (length) of the vector.
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Normalizes the vector (makes its length equal to 1).
    pub fn normalize(&self) -> Self {
        let length = self.length();
        if length == 0.0 {
            Self::ZERO
        } else {
            *self / length
        }
    }

    /// Calculates the dot product of this vector and another.
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Returns the squared length of the vector (avoids sqrt for performance).
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Performs linear interpolation between two vectors.
    pub fn lerp(start: &Self, end: &Self, t: f32) -> Self {
        *start + (*end - *start) * t
    }

    /// Rotates the vector by the given angle (in radians).
    pub fn rotate(&self, angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            x: self.x * cos - self.y * sin,
            y: self.x * sin + self.y * cos,
        }
    }

    /// Calculates the distance between this vector and another.
    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Operator overloads for Vec2
impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

/// Additional operator overloads for scalar multiplication.
impl Mul<Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::Vec2;

    #[test]
    fn test_creation() {
        let vec = Vec2::new(3.0, 4.0);
        assert_eq!(vec.x, 3.0);
        assert_eq!(vec.y, 4.0);
    }

    #[test]
    fn test_length() {
        let vec = Vec2::new(3.0, 4.0);
        assert_eq!(vec.length(), 5.0);
    }

    #[test]
    fn test_normalize() {
        let vec = Vec2::new(3.0, 4.0).normalize();
        assert!((vec.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let vec1 = Vec2::new(1.0, 0.0);
        let vec2 = Vec2::new(0.0, 1.0);
        assert_eq!(vec1.dot(&vec2), 0.0);
    }

    #[test]
    fn test_lerp() {
        let start = Vec2::new(0.0, 0.0);
        let end = Vec2::new(10.0, 10.0);
        let result = Vec2::lerp(&start, &end, 0.5);
        assert_eq!(result, Vec2::new(5.0, 5.0));
    }

    #[test]
    fn test_rotation() {
        let vec = Vec2::new(1.0, 0.0);
        let rotated = vec.rotate(std::f32::consts::PI / 2.0);
        assert!((rotated.x - 0.0).abs() < 1e-6);
        assert!((rotated.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance() {
        let vec1 = Vec2::new(0.0, 0.0);
        let vec2 = Vec2::new(3.0, 4.0);
        assert_eq!(vec1.distance(&vec2), 5.0);
    }
}
