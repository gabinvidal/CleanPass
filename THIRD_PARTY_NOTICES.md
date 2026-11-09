# Third-party notices

CleanPass redistributes the runtime of:

- **Intel® Open Image Denoise 2.5.1**, Copyright 2018 Intel Corporation, Apache License 2.0.
  https://github.com/RenderKit/oidn
- **oneAPI Threading Building Blocks**, shipped with OIDN, Apache License 2.0.

`build.ps1` gives this runtime private file names (`OIDN_CleanPass25*.dll`,
`tbbCP.dll`) and renames the matching import names inside the DLLs (same
length, no other change), so that it cannot clash with the libraries Nuke or
another plugin load. `MSVCP_CP.dll` is a renamed, unmodified copy of the
Microsoft Visual C++ runtime `msvcp140.dll`, taken at build time from the
Visual Studio / Visual C++ Redistributable install of the machine that builds
it (Nuke ships an older one); it is a redistributable file under the Microsoft
Visual C++ Redistributable licence terms.

The full licence texts and third-party notices are in `oidn/` (copied from the
OIDN SDK by `build.ps1`). Nuke and the Nuke NDK are © The Foundry and are not
redistributed.
