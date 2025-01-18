use emerge_math::{Vec2, Rect, Circle, Triangle};

fn main() {
    // Vec2
    let vec = Vec2::new(3.0, 4.0);
    println!("Vector: {:?}", vec);
    println!("Length: {}", vec.length());

    // Rect
    let rect = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(4.0, 5.0));
    println!("Rectangle: {:?}", rect);
    println!("Area: {}", rect.area());
    println!(
        "Does rect contain {:?}? {}",
        Vec2::new(2.0, 2.0),
        rect.contains(Vec2::new(2.0, 2.0))
    );
    println!(
        "Does rect contain {:?}? {}",
        Vec2::new(5.0, 5.0),
        rect.contains(Vec2::new(5.0, 5.0))
    );

    let rect2 = Rect::new(Vec2::new(2.0, 2.0), Vec2::new(3.0, 3.0));
    println!("Another Rectangle: {:?}", rect2);
    println!(
        "Do the rectangles intersect? {}",
        rect.intersects(&rect2)
    );

    if let Some(intersection) = rect.intersection(&rect2) {
        println!("Intersection: {:?}", intersection);
    } else {
        println!("No intersection found.");
    }

    // Circle
    let circle = Circle::new(Vec2::new(0.0, 0.0), 5.0);
    println!("Circle: {:?}", circle);
    println!(
        "Does circle contain {:?}? {}",
        Vec2::new(3.0, 4.0),
        circle.contains(&Vec2::new(3.0, 4.0))
    );

    let another_circle = Circle::new(Vec2::new(7.0, 0.0), 3.0);
    println!(
        "Do the circles intersect? {}",
        circle.intersects(&another_circle)
    );

    // Triangle
    let tri = Triangle::new(
        Vec2::new(0.0, 0.0),
        Vec2::new(4.0, 0.0),
        Vec2::new(0.0, 3.0),
    );
    println!("Triangle: {:?}", tri);
    println!("Area: {}", tri.area());
    println!(
        "Does triangle contain {:?}? {}",
        Vec2::new(1.0, 1.0),
        tri.contains(&Vec2::new(1.0, 1.0))
    );

    let other_tri = Triangle::new(
        Vec2::new(2.0, 1.0),
        Vec2::new(5.0, 1.0),
        Vec2::new(2.0, 4.0),
    );
    println!(
        "Do triangles intersect? {}",
        tri.intersects(&other_tri)
    );
}
