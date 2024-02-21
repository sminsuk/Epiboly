""" Global particle and bond data. For storing data associated with TF objects (Particles, Bonds, Angles)

But, more generally, storing system state. So storing visibility here as well.

Maybe put this in a class?
Usage:
    Note: The original uses of the Bond and Particle dictionaries were to store data that was not retrievable from
    individual bonds and particles because of bugs in Tissue Forge. Later, once those bugs were fixed, these
    structures were not storing anything useful. However, this was good working infrastructure, so I kept it
    in place for future use.
    
    Angles were added later, and that was also for a bug, which likely still exists in v0.1.0. (But that one
    never needed actual content, just keys, as described below.) If those anomalous Angle bonds get fixed (prevented)
    in a future release, then won't need this for Angles.
    
    And now the Particle dictionary has a new use: the cell_radius value that needs to be associated with each
    particle. TF objects are not customizable in that way; you cannot give Particles (or ParticleHandles) new
    properties. So, store the value here, and look it up when needed, by ParticleId.
    
    Use the convenience functions below to create Bonds and Angles, and to add newly created Particles to the
    catalog (or to update existing ones); and to destroy any of those. Alternatively:
    
    Whenever creating a particle / bond / angle, add it to particles_by_id / bonds_by_id / angles_by_id, respectively
    Whenever destroying a particle / bond / angle, delete it from those dicts
    
    BondData currently empty.
    ParticleData now contains one key, "cell_radius".
    AngleData never needed any content: the idea is to identify anomalous Angle objects created due to a
        TF bug, by the fact that they are NOT in the dict. All we really need is the keys; using id as placeholder
        value. (We'll never need to look up the value.)
    
    Whenever a particle .becomes(LeadingEdge), set its visibility to True;
    Whenever a particle .becomes(Little) (interior particle), set its visibility according to the state flag here.
        (For now, change of state of the flag itself happens in module "interactive".)
"""
from typing import TypedDict

import tissue_forge as tf
import utils.tf_utils as tfu

class BondData(TypedDict, total=True):
    spring_constant: float

class ParticleData(TypedDict, total=True):
    cell_radius: float

bonds_by_id: dict[int, BondData] = {}
angles_by_id: dict[int, int] = {}
particles_by_id: dict[int, ParticleData] = {}

def create_bond(potential: tf.Potential, k: float, p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> tf.BondHandle:
    """Create the bond, and store k for future retrieval, because the value in the potential isn't accessible"""
    handle: tf.BondHandle = tf.Bond.create(potential, p1, p2)
    bond_values: BondData = {"spring_constant": k}
    bonds_by_id[handle.id] = bond_values
    return handle

def get_spring_constant(bhandle: tf.BondHandle) -> float:
    """Return spring constant for a given bond"""
    assert bhandle.id in bonds_by_id, f"Bond {bhandle.id} not in dictionary!"
    return bonds_by_id[bhandle.id]["spring_constant"]

def destroy_bond(bhandle: tf.BondHandle) -> None:
    del bonds_by_id[bhandle.id]
    bhandle.destroy()
    
def create_angle(potential: tf.Potential,
                 outer_p1: tf.ParticleHandle,
                 center_p: tf.ParticleHandle,
                 outer_p2: tf.ParticleHandle) -> tf.AngleHandle:
    handle: tf.AngleHandle = tf.Angle.create(potential, outer_p1, center_p, outer_p2)
    angles_by_id[handle.id] = handle.id
    return handle

def destroy_angle(angle_handle: tf.AngleHandle) -> None:
    del angles_by_id[angle_handle.id]
    angle_handle.destroy()
    
def add_particle(phandle: tf.ParticleHandle, radius: float) -> None:
    """ 'Add' rather than 'Create' because TF routines handle the creation part
    
    This can also be used to update the stored data when the cell_radius changes.
    """
    particle_values: ParticleData = {"cell_radius": radius}
    particles_by_id[phandle.id] = particle_values

def destroy_particle(phandle: tf.ParticleHandle) -> None:
    # If particle has any bonds, destroy them. TF would do that, but would not remove them from the catalog
    b: tf.BondHandle
    for b in tfu.bonds(phandle):
        destroy_bond(b)
        
    del particles_by_id[phandle.id]
    phandle.destroy()
    
def get_cell_radius(phandle: tf.ParticleHandle) -> float:
    """Return cell_radius for a given particle
    
    ToDo maybe someday: To do this "right", I should have a get_attr() function where you pass the attribute name.
    Then it would be general for any arbitrary attribute. But I don't want to have to type a string in the function
    call. I would want it to be a token that the IDE recognizes and can code-complete. Would have to think about how
    to get python to do that. If I ever have tons of different attributes to access, maybe then. Right now I
    only have this one attribute, so this is good enough!
    """
    assert phandle.id in particles_by_id, f"Particle {phandle.id} not in dictionary!"
    return particles_by_id[phandle.id]["cell_radius"]

def set_cell_radius(phandle: tf.ParticleHandle, radius: float) -> None:
    """Update the stored radius for a particle
    
    For now, this is actually just an alias for add_particle(), but semantically it makes more sense to use
    this when updating a particle as opposed to creating a new one.
    
    If ParticleData is later enhanced with additional attributes, the two will no longer be identical.
    """
    # Just a pass-through. When particle does not yet exist in the table, add_particle() adds it. When it
    # already exists, add_particle() overwrites it.
    add_particle(phandle, radius)

def initialize_state() -> None:
    """Recreates the global catalogs based on imported data from a previous run
    
    Currently the Bond and Angle dicts contain no meaninful data, but we still need to create them and fill them
    with dummy values, so that when existing items are destroyed, their dict entries can be del'ed.
    
    The Particle dict does contain meaningful data, but the ids are now incorrect, so we have to rebuild it.
    Imported particle ids won't be the same as when they were exported, but TF provides a mapping that
    will disappear at the first time step. Thus, this function MUST be called AFTER state import and BEFORE
    any stepping.
    """
    global particles_by_id
    
    for bhandle in tf.BondHandle.items():
        bond_values: BondData = {}
        bonds_by_id[bhandle.id] = bond_values
        
    for handle in tf.AngleHandle.items():
        angles_by_id[handle.id] = handle.id
        
    # Create new dictionary with new keys mapped from the old keys; replace old dictionary with new dictionary.
    # Note, json exported int keys are turned into strings, so we have to change them back!
    particles_by_id = {tf.io.mapImportParticleId(int(old_key)): old_value
                       for old_key, old_value in particles_by_id.items()}
    
    # mapImportParticleId() returns -1 if the old id is not found
    assert -1 not in particles_by_id, "One or more exported ids were not found in the imported data!"
        
def clean_state() -> None:
    """Clean out all anomalous Angle bonds at once
    
    Because of the bug that required this dict to exist in the first place, we need a function to purge any
    anomalous angles that should not exist, BEFORE saving simulation state.
    """
    angle: tf.AngleHandle
    for angle in tf.AngleHandle.items():
        if angle.id not in angles_by_id:
            print(tfu.bluecolor + f"Destroying false angle (id={angle.id})" + tfu.endcolor)
            angle.destroy()

visibility_state: bool = True

def get_state() -> dict:
    """generate state to be saved to disk"""
    return {"particles_by_id": particles_by_id}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global particles_by_id
    particles_by_id = d["particles_by_id"]
