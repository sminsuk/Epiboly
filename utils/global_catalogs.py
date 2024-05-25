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
    
    And now the Particle dictionary has a new use: the cell radius value that needs to be associated with each
    particle. TF objects are not customizable in that way; you cannot give Particles (or ParticleHandles) new
    properties. So, store the value here, and look it up when needed, by ParticleId.
    
    And, lt = lineage tracer, a flag for cell labeling.
    
    Similarly, the Bond dictionary has a new use: the spring constant. This is the value k in tf.Potential.harmonic
    objects, but unfortunately it is not readable. I now have bonds having different spring constants, and I
    need to be able to retrieve them, so that means I have to store them in the catalog for every bond I create.
    
    The dictionary fields for these values are named in a highly abbreviated fashion (r, lt, k) in order to keep
    file size reasonable, because thousands of copies of each name (one for each particle) are saved in the
    serialized output file.
    
    Use the convenience functions below to create Bonds and Angles, and to add newly created Particles to the
    catalog (or to update existing ones); and to destroy any of those. Alternatively:
    
    Whenever creating a particle / bond / angle, add it to particles_by_id / bonds_by_id / angles_by_id, respectively
    Whenever destroying a particle / bond / angle, delete it from those dicts
    
    BondData now contains one key, the spring constant "k".
    ParticleData now contains two keys, the cell radius "r", and the lineage tracer flag "lt".
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
    k: float    # spring constant

class ParticleData(TypedDict, total=True):
    r: float    # cell radius
    lt: bool    # lineage tracer

bonds_by_id: dict[int, BondData] = {}
angles_by_id: dict[int, int] = {}
particles_by_id: dict[int, ParticleData] = {}

def create_bond(potential: tf.Potential, k: float, p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> tf.BondHandle:
    """Create the bond, and store k for future retrieval, because the value in the potential isn't accessible"""
    handle: tf.BondHandle = tf.Bond.create(potential, p1, p2)
    bond_values: BondData = {"k": k}
    bonds_by_id[handle.id] = bond_values
    return handle

def get_spring_constant(bhandle: tf.BondHandle) -> float:
    """Return spring constant for a given bond"""
    assert bhandle.id in bonds_by_id, f"Bond {bhandle.id} not in dictionary!"
    return bonds_by_id[bhandle.id]["k"]

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
    
    Lineage tracer flag will always be created as False
    """
    particle_values: ParticleData = {"r": radius,
                                     "lt": False}
    particles_by_id[phandle.id] = particle_values
    
def copy_particle(copy: tf.ParticleHandle, original: tf.ParticleHandle) -> None:
    """Copy all particle attributes from one particle to another
    
    This can be used either to add a new particle to the dictionary, or to modify an existing one,
    so long as you want to overwrite all attribute values on the copy, from the original
    """
    assert original.id in particles_by_id, f"Particle {original.id} not in dictionary!"
    particles_by_id[copy.id] = particles_by_id[original.id]

def destroy_particle(phandle: tf.ParticleHandle) -> None:
    # If particle has any bonds, destroy them. TF would do that, but would not remove them from the catalog
    b: tf.BondHandle
    for b in tfu.bonds(phandle):
        destroy_bond(b)
        
    del particles_by_id[phandle.id]
    phandle.destroy()
    
def get_cell_radius(phandle: tf.ParticleHandle) -> float:
    """Return cell radius for a given particle
    
    ToDo maybe someday: To do this "right", I should have a get_attr() function where you pass the attribute name.
    Then it would be general for any arbitrary attribute. But I don't want to have to type a string in the function
    call. I would want it to be a token that the IDE recognizes and can code-complete. Would have to think about how
    to get python to do that. If I ever have tons of different attributes to access, maybe then. Right now I
    only have very few attributes (this one and lt), so this is good enough!
    """
    assert phandle.id in particles_by_id, f"Particle {phandle.id} not in dictionary!"
    return particles_by_id[phandle.id]["r"]

def get_lineage_tracer(phandle: tf.ParticleHandle) -> bool:
    """Return lineage tracer status for a given particle"""
    assert phandle.id in particles_by_id, f"Particle {phandle.id} not in dictionary!"
    return particles_by_id[phandle.id]["lt"]

def set_cell_radius(phandle: tf.ParticleHandle, radius: float) -> None:
    """Update the stored radius for a particle"""
    assert phandle.id in particles_by_id, f"Particle {phandle.id} not in dictionary!"
    attributes: ParticleData = particles_by_id[phandle.id]
    attributes["r"] = radius
    particles_by_id[phandle.id] = attributes
    
def set_cell_lineage_tracer(phandle: tf.ParticleHandle) -> None:
    """Update the stored lineage tracer flag for a particle, to True
    
    You can never take lineage tracer away, you can only add it. Cells always start as False;
    this makes it True. The cell is now labeled.
    """
    attributes: ParticleData = particles_by_id[phandle.id]
    attributes["lt"] = True
    particles_by_id[phandle.id] = attributes

def initialize_state() -> None:
    """Recreates the global catalogs based on imported data from a previous run
    
    Currently the Angle dict contains no meaninful data, but we still need to create it and fill it
    with dummy values, so that when existing items are destroyed, their dict entries can be del'ed.
    
    The Particle dict does contain meaningful data, but the ids are now incorrect, so we have to rebuild it.
    Imported particle ids won't be the same as when they were exported, but TF provides a mapping that
    will disappear at the first time step. Thus, this function MUST be called AFTER state import and BEFORE
    any stepping.
    
    The Bond dict also contains meaningful data, and like Particles, their ids have changed. However, TF
    documentation implies that's not the case! And provides no mechanism for retrieving the new ones.
    So for right now, re-import is completely broken. I have filed an issue:
    https://github.com/tissue-forge/tissue-forge/issues/65
    ToDo: Fix this!
    """
    global particles_by_id, bonds_by_id
    
    for handle in tf.AngleHandle.items():
        angles_by_id[handle.id] = handle.id
        
    # Create new dictionary with new keys mapped from the old keys; replace old dictionary with new dictionary.
    # Note, json exported int keys are turned into strings, so we have to change them back!
    particles_by_id = {tf.io.mapImportParticleId(int(old_key)): old_value
                       for old_key, old_value in particles_by_id.items()}
    
    # mapImportParticleId() returns -1 if the old id is not found
    assert -1 not in particles_by_id, "One or more exported ids were not found in the imported data!"
    
    # If Bond ids were correct, as implied by the TF docs, the following would work. Instead,
    # the stored bond ids do not match the imported ones! ToDO: Fix this!
    # Regardless, need to turn the str keys back to integers.
    bonds_by_id = {int(old_key): old_value
                   for old_key, old_value in bonds_by_id.items()}
        
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
    return {"particles_by_id": particles_by_id,
            "bonds_by_id": bonds_by_id}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global particles_by_id, bonds_by_id
    particles_by_id = d["particles_by_id"]
    bonds_by_id = d["bonds_by_id"]
