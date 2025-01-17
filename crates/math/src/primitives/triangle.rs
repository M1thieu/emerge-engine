use super::vec2::Vec2;

/// A structure representing a triangle in 2D space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    pub a: Vec2,
    pub b: Vec2,
    pub c: Vec2,
}

impl Triangle {
    /// Creates a new triangle given its three vertices.
    pub fn new(a: Vec2, b: Vec2, c: Vec2) -> Self {
        Self { a, b, c }
    }

    /// Calculates the area of the triangle using the shoelace formula.
    pub fn area(&self) -> f32 {
        ((self.a.x * (self.b.y - self.c.y)
            + self.b.x * (self.c.y - self.a.y)
            + self.c.x * (self.a.y - self.b.y))
            .abs())
            / 2.0
    }

    /// Checks if a point is inside the triangle using barycentric coordinates.
    pub fn contains(&self, point: &Vec2) -> bool {
        let area_total = self.area();
        let area1 = Triangle::new(*point, self.b, self.c).area();
        let area2 = Triangle::new(self.a, *point, self.c).area();
        let area3 = Triangle::new(self.a, self.b, *point).area();

        // Check if the sum of sub-areas equals the total area (with a small epsilon for floating-point errors)
        (area1 + area2 + area3 - area_total).abs() < f32::EPSILON
    }

    /// Checks if the triangle intersects with another triangle.
    pub fn intersects(&self, other: &Triangle) -> bool {
        // Check if any vertex of one triangle is inside the other triangle.
        self.contains(&other.a) || self.contains(&other.b) || self.contains(&other.c)
            || other.contains(&self.a) || other.contains(&self.b) || other.contains(&self.c)
    }
}

#[cfg(test)]
mod tests {
    use super::Triangle;
    use super::Vec2;

    #[test]
    fn test_area() {
        let tri = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        assert_eq!(tri.area(), 6.0);
    }

    #[test]
    fn test_contains() {
        let tri = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        assert!(tri.contains(&Vec2::new(1.0, 1.0)));
        assert!(!tri.contains(&Vec2::new(5.0, 5.0)));
    }

    #[test]
    fn test_intersects() {
        let tri1 = Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        let tri2 = Triangle::new(
            Vec2::new(2.0, 1.0),
            Vec2::new(5.0, 1.0),
            Vec2::new(2.0, 4.0),
        );
        assert!(tri1.intersects(&tri2));

        let tri3 = Triangle::new(
            Vec2::new(5.0, 5.0),
            Vec2::new(6.0, 5.0),
            Vec2::new(5.0, 6.0),
        );
        assert!(!tri1.intersects(&tri3));
    }
}
