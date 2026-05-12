//! Partial Differential Equation solvers (CPU-only finite differences)
//! ====================================================================
//!
//! Generic PDE primitives:
//!
//! - [`fokker_planck`] — 1-D forward Fokker–Planck (Kolmogorov forward) solver
//!   with conservative central differences.
//! - [`hjb_multid`] — explicit upwind scheme for multidimensional
//!   Hamilton–Jacobi–Bellman equations on a regular Cartesian grid.
//! - [`elliptic_fd`] — Jacobi/Gauss–Seidel iteration for the Poisson equation
//!   `-Δu = f` with Dirichlet boundary conditions on a 2-D rectangle.

pub mod fokker_planck;
pub mod hjb_multid;
pub mod elliptic_fd;

pub use fokker_planck::{FokkerPlanckConfig, FokkerPlanckResult, solve_fokker_planck_1d};
pub use hjb_multid::{HjbMultidConfig, HjbMultidResult, solve_hjb_multid};
pub use elliptic_fd::{EllipticFdConfig, EllipticFdResult, solve_poisson_2d};
