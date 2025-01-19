use nalgebra::{Vector2 as Vec2, Point2, Matrix3, Rotation2};

fn main() {
    // Vec2 operations
    let vec = Vec2::new(3.0, 4.0);
    println!("Vector: {:?}", vec);
    println!("Norm (Length): {}", vec.norm()); // Use norm() for vector length

    let normalized_vec = vec.normalize();
    println!("Normalized Vector: {:?}", normalized_vec);

    // Point2 operations
    let point1 = Point2::new(0.0, 0.0);
    let point2 = Point2::new(3.0, 4.0);
    println!("Point 1: {:?}", point1);
    println!("Point 2: {:?}", point2);

    let distance = nalgebra::distance(&point1, &point2); // Compute distance between points
    println!("Distance between points: {}", distance);

    // Scaling using a scalar
    let scale_factor = 2.0;
    let scaled_vec = vec * scale_factor;
    println!("Scaled Vector: {:?}", scaled_vec);

    // Rotation using Rotation2
    let rotation = Rotation2::new(std::f32::consts::PI / 4.0); // Rotate by 45 degrees
    let rotated_vec = rotation * vec;
    println!("Rotated Vector: {:?}", rotated_vec);

    // Combining transformations using Matrix3
    let scaling_matrix = Matrix3::new(
        2.0, 0.0, 0.0,  // Scale x by 2
        0.0, 2.0, 0.0,  // Scale y by 2
        0.0, 0.0, 1.0,  // Homogeneous coordinate
    );
    let rotation_matrix = rotation.to_homogeneous();
    let combined_transform = rotation_matrix * scaling_matrix;
    println!("Combined Transformation Matrix: {:?}", combined_transform);

    // Applying the transformation to a point
    let transformed_point = combined_transform.transform_point(&point1);
    println!("Transformed Point: {:?}", transformed_point);
}
