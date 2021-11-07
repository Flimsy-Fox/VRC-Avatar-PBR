# VRC-Avatar-PBR
A PBR-based shading solution for VRChat avatars.
===========

This shader, at the time being, takes the following information for use in PBR shading:
- The most dominant reflection cubemap
- The most dominant light source
- The skybox color

There are some additional features available to this shader, main of which is AudioLink, tied to the emission map.

In the short term, I hope to acheive the following with this shader:
~~- Casting onto reflection, light, and shadow maps~~
- First-bounce ray casting onto the shader for more accurate lighting
- Implementing a shader properties addon

I am tentative about adding the following features to this shader:
- Displacement map support
- Refraction
- Transmission
