# GenerateDynamicalNetworks

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://denizhanpak.github.io/GenerateDynamicalNetworks.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://denizhanpak.github.io/GenerateDynamicalNetworks.jl/dev/)
[![Build Status](https://github.com/denizhanpak/GenerateDynamicalNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/denizhanpak/GenerateDynamicalNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Given a transition graph and the mean time spent in each state, build a noisy dynamical system whose
switching between states reproduces both. The system can be built with several constructions:

- Lotka–Volterra (Horchler)
- simplex and cylinder
- two-population
- CTRNN
- phase oscillators
- Hopfield

The dynamical class of the result is measured and reported: heteroclinic, excitable, ghost or limit cycle.

**Status:** under construction. The code is being ported from the research prototype in
[`research/`](research/), and the package doesn't have its API yet.

## License

MIT; see [LICENSE](LICENSE).
