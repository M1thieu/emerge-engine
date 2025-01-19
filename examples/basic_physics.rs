use rapier2d::prelude::*;

fn main() {
    // 1. Create the physics world
    let mut physics_pipeline = PhysicsPipeline::new();
    let gravity = vector![0.0, -9.81]; // Gravity vector pointing downward
    let mut island_manager = IslandManager::new();
    let mut broad_phase = BroadPhaseMultiSap::new(); // Use the concrete type
    let mut narrow_phase = NarrowPhase::new();       // Use the concrete type
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();
    let mut ccd_solver = CCDSolver::new();
    let mut query_pipeline = QueryPipeline::new();

    let integration_parameters = IntegrationParameters::default();

    // 2. Add a ground body (static body)
    let ground_body = RigidBodyBuilder::fixed().translation(vector![0.0, -10.0]).build();
    let ground_handle = bodies.insert(ground_body);

    let ground_collider = ColliderBuilder::cuboid(50.0, 1.0).build();
    colliders.insert_with_parent(ground_collider, ground_handle, &mut bodies);

    // 3. Add a dynamic body (falling box)
    let box_body = RigidBodyBuilder::dynamic().translation(vector![0.0, 10.0]).build();
    let box_handle = bodies.insert(box_body);

    let box_collider = ColliderBuilder::cuboid(1.0, 1.0).density(1.0).build();
    colliders.insert_with_parent(box_collider, box_handle, &mut bodies);

    // 4. Run the simulation for a few steps
    for step in 0..100 {
        // Step the physics simulation
        physics_pipeline.step(
            &gravity,
            &integration_parameters,
            &mut island_manager,
            &mut broad_phase,
            &mut narrow_phase,
            &mut bodies,
            &mut colliders,
            &mut impulse_joints,
            &mut multibody_joints,
            &mut ccd_solver,
            Some(&mut query_pipeline),
            &(),
            &(),
        );

        // Print the position of the dynamic body
        let box_body = bodies.get(box_handle).unwrap();
        println!(
            "Step {}: Box position: {:?}",
            step,
            box_body.translation()
        );
    }
}