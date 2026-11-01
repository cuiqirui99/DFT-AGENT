# Replacing ASE

DFT Agent uses ASE 3.29.0 under LGPL-2.1-or-later. You may inspect, modify and
replace ASE, and debug those changes, as permitted by its license. The DFT
Agent license preserves these rights. The full LGPL text is in
`licenses/LGPL-2.1.txt`; source is supplied in the companion third-party source
archive described in `THIRD-PARTY-SOURCES.md`.

ASE is loaded from ordinary Python files outside the application's compiled
archive. No Python installation is needed to edit those files.

1. Quit DFT Agent. Make a copy of `DFT Agent.app` in a folder you can write to.
2. In Finder, right-click the copy and choose **Show Package Contents**.
3. Open `Contents/Resources/ase`. Replace or edit the compatible ASE files.
   The runtime may report the equivalent `Contents/Frameworks/ase` path;
   the bundle links these locations to the same files.
4. Remove any `__pycache__` folders inside the replaced ASE directory so old
   cached bytecode cannot hide your changes.
5. Changing application files invalidates their original signature. To sign
   your private modified copy locally, run this in Terminal, using its path:

   ```sh
   codesign --force --deep --sign - "/path/to/DFT Agent.app"
   ```

6. Open the modified copy. Keep the original application as a backup.

Use the supplied ASE 3.29.0 source as a starting point and preserve its package
layout. A different ASE release must remain API-compatible with the application;
an incompatible replacement can fail to start or change scientific results.
The bundled Python interpreter is Python 3.13.

The replacement mechanism was checked by changing ASE's initialization module
in a copied application, re-signing that copy, and verifying that the running
application imported the modified external module. The original application
was not changed.

These instructions do not grant rights to unrelated proprietary source code
or remove the licenses of any other component. Your rights under LGPL and
other applicable third-party licenses remain unchanged.
