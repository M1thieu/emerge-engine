use super::vec2::Vec2;

/// A rectangle primitive, represented by a position and size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub position: Vec2,
    pub size: Vec2,
}

impl Rect {
    /// Creates a new rectangle with the given position and size.
    pub fn new(position: Vec2, size: Vec2) -> Self {
        Self { position, size }
    }

    /// Calculates the area of the rectangle.
    pub fn area(&self) -> f32 {
        self.size.x * self.size.y
    }

    /// Checks if a point is inside the rectangle.
    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.position.x
            && point.x <= self.position.x + self.size.x
            && point.y >= self.position.y
            && point.y <= self.position.y + self.size.y
    }

    /// Checks if this rectangle intersects with another rectangle.
    pub fn intersects(&self, other: &Rect) -> bool {
        !(self.position.x + self.size.x <= other.position.x
            || other.position.x + other.size.x <= self.position.x
            || self.position.y + self.size.y <= other.position.y
            || other.position.y + other.size.y <= self.position.y)
    }

    /// Returns the intersection of this rectangle with another rectangle, if any.
    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        if !self.intersects(other) {
            return None;
        }

        let x1 = self.position.x.max(other.position.x);
        let y1 = self.position.y.max(other.position.y);
        let x2 = (self.position.x + self.size.x).min(other.position.x + other.size.x);
        let y2 = (self.position.y + self.size.y).min(other.position.y + other.size.y);

        Some(Rect::new(Vec2::new(x1, y1), Vec2::new(x2 - x1, y2 - y1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_area() {
        let rect = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(4.0, 5.0));
        assert_eq!(rect.area(), 20.0);
    }

    #[test]
    fn test_contains() {
        let rect = Rect::new(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
        assert!(rect.contains(Vec2::new(2.0, 2.0)));
        assert!(!rect.contains(Vec2::new(0.0, 0.0)));
    }

    #[test]
    fn test_intersects() {
        let rect1 = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(4.0, 4.0));
        let rect2 = Rect::new(Vec2::new(2.0, 2.0), Vec2::new(4.0, 4.0));
        assert!(rect1.intersects(&rect2));

        let rect3 = Rect::new(Vec2::new(5.0, 5.0), Vec2::new(2.0, 2.0));
        assert!(!rect1.intersects(&rect3));
    }

    #[test]
    fn test_intersection() {
        let rect1 = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(4.0, 4.0));
        let rect2 = Rect::new(Vec2::new(2.0, 2.0), Vec2::new(4.0, 4.0));
        let expected = Rect::new(Vec2::new(2.0, 2.0), Vec2::new(2.0, 2.0));
        assert_eq!(rect1.intersection(&rect2), Some(expected));

        let rect3 = Rect::new(Vec2::new(5.0, 5.0), Vec2::new(2.0, 2.0));
        assert_eq!(rect1.intersection(&rect3), None);
    }
}
