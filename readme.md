AWGN
====

Simple crate with minimal dependencies to provide the random number generators
that I actually want on a regular basis. The family of
[rand](https://crates.io/crates/rand) crates are fantastic, but pretty bloated
if all you ever want is U[0, 1), N(0, 1), and occasionally CN(0, 2).


TODO
----

- Currently I have a `Generator` object which just has all the methods inside
  it, I think I'll probably break that into a `Uniform` and a `Gaussian`.
- Provide some utilities for scaling particularly the Gaussian distributions to
  achieve particular signal to noise ratios.
