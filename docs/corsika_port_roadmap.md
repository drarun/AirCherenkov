# CORSIKA Feature Port – Roadmap

Items to investigate porting from CORSIKA into our fully-tensorized
GPU simulation chain.  These are "v2.0" improvements that do **not**
block the current Crab Nebula spectral reconstruction.

## High Priority (affects IRFs quantitatively)
- [ ] **Layered atmospheric density model** – CORSIKA uses a 5-layer
      exponential atmosphere (US Standard 1976).  We currently use a
      single exponential.  Affects Cherenkov yield vs altitude and
      therefore the effective-area curve shape, especially at large
      zenith angles.
- [ ] **Wavelength-dependent Cherenkov emission & detector response** –
      mirror reflectivity and PMT quantum efficiency are functions of
      wavelength (peak ~320 nm).  CORSIKA tracks λ per photon; we use
      scalar efficiencies.

## Medium Priority (affects gamma/hadron separation)
- [ ] **Improved hadronic interaction model** – our proton cascade is
      simplified (fixed π⁰ / π± / K ratios).  CORSIKA offers QGSJET-II,
      SIBYLL, EPOS-LHC.  A lookup-table approach could approximate the
      multiplicity/inelasticity distributions without a full MC.
- [ ] **Muon propagation** – we currently stop muons after creation.
      CORSIKA propagates them to the ground (important for muon-ring
      identification in hadron rejection).

## Lower Priority (nice-to-have)
- [ ] **Curved atmosphere / Earth curvature** – matters above ~60° zenith.
- [ ] **Refractive index altitude profile** – Cherenkov angle varies with
      atmospheric density; we use a constant n = 1.0003.
- [ ] **Geomagnetic field model** – we have a constant B-field; CORSIKA
      uses IGRF coefficients for the observation epoch.
