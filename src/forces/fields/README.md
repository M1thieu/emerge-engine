# fields

**Level:** law. **Status:** the explicit solver's substep hook; on the precise path only
gravity is carried so far, and the path refuses other fields, with the reason.

Body forces acting on matter from outside: gravity and n-body gravity, Coulomb and
uniform electric fields, the gyroscopic pair (uniform magnetic, Coriolis), buoyancy,
drag. The `Field` trait is defined in `mod.rs`.

**Derives from:** the force laws, cited in each file. Depends only on the particle, never
the reverse.
