"""Have these three values be global by importing this wherever needed

Trying to have the simplest possible access to these three variables, which I reference hundreds and hundreds
of times.

Module epiboly_init was supposed to achieve that, and did for awhile; I simply could:
from epiboly_init import LeadingEdge
which provided basically global symbols. But once I added the ability to restart a sim by loading saved state,
things got a lot more complicated. epiboly_init had to import additional other modules, leading to module import
circularity. There was no way to get around it, and still keep the simple naming, so finally had to resort to
referencing the variables through a module name. Could do that with epiboly_init, but for clarity and separation of
concerns, putting them in their own module. And this one should not have to import anything other than tissue-forge.

This means, they'll be unprotected. Note that only epiboly_init is allowed to write to them. For everybody
else, these are treated as read only.

Apparently, I don't need to create actual values here; the code to do that can remain in epiboly_init.
Surprising because as uninitialized type hints, they don't really exist as far as the runtime is concerned!
But I guess even just naming them, is a runtime python statement that creates storage.
"""
import tissue_forge as tf

Big: tf.ParticleType
Little: tf.ParticleType
LeadingEdge: tf.ParticleType
