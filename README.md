# VRC-Avatar-PBR
A PBR-based shading solution for VRChat avatars.
===========

Please support me on Gumroad! [Let's go!](https://flimsyfox.gumroad.com/l/VRC-PBR)

NOTE: Shader may not appear correctly when Vulkan (the default graphics API on AMD GPUs) is being used. This issue is currently being resolved.
----------------------------

This shader, at the time being, takes the following information for use in PBR shading:
- Reflection maps
- Light maps
- The skybox color

There are some additional features available to this shader, main of which is AudioLink, tied to the emission map.

In the short term, I hope to acheive the following with this shader:

- Casting onto reflection, light, and shadow maps (DONE)
- First-bounce ray casting onto the shader for more accurate lighting
- Implementing a shader properties addon (DONE)

I am tentative about adding the following features to this shader:
- Displacement map support (DONE?)
- Refraction
- Transmission
