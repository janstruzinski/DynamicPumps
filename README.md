# DynamicPumps

## Introduction

Hi,

Welcome to DynamicPumps, a Python library for sizing and analysis of centrifugal pumps and axial inducers. It was
created to quickly size impellers and volutes based on on-design point requirements, as well as to create preliminary
performance maps. I hope you will find it useful.

With kind regards, Jan

## General Overview

DynamicPumps provides objects for sizing hydraulic components of pumps and a class to gather preliminary performance
data for single- or multi- stage pump system. A pump is first sized at a design point and may then be analysed at
other mass flow rates and rotational speeds. Sizing results are stored in stage objects, which can be used by
the system class. This class passes the outlet state of one stage to the inlet of the next, adds stage head,
pressure rise and power, and can generate a performance map over prescribed mass-flow and speed ranges.

The repository is organised as follows:

- `DynamicPumps/BarskePump.py` defines the `BarskePump` geometry, sizing, analysis, verification, reporting and
  geometry-plotting methods.
- `DynamicPumps/PumpSystem.py` defines `PumpSystem`, which analyses an ordered list of stages and generates pump system
performance maps.
- `DynamicPumps/Fluid.py` defines the `Fluid` property interface used by the hydraulic models.
- `DynamicPumps/functions.py` converts between pressure rise and head for constant- or variable-density fluids.
- `DynamicPumps/example.py` demonstrates sizing one Barske pump, placing it in a `PumpSystem`, sweeping mass flow and
  speed, and plotting a map.

The current implemented hydraulic stage is `BarskePump`, but full emission impellers and axial inducers will be added
in the future. `PumpSystem` is stage-oriented: pump stages can be added to its ordered
`stages` list when they provide the same `analyse(fluid, mdot, RPM, p_upstream, T_upstream)` interface and result
outputs.

All dimensional inputs and outputs use SI units unless a method description states otherwise. Pressure-like inputs
are in Pa, temperature is in K, mass flow is in kg/s, rotational speed is in rpm and lengths are in m.

## Installation

Install the package directly from GitHub:

```text
pip3 install git+https://github.com/janstruzinski/DynamicPumps
```

## Disclaimer

I wrote this whole library by myself. LLM was only used for the generation of this README.

## Documentation

### BarskePump

`BarskePump` in `DynamicPumps/BarskePump.py` represents a partial-emission centrifugal pump. Its principal workflow is:

1. Construct an empty `BarskePump` object.
2. Call `size_dimensions(...)` once at the design point. This calculates and stores the pump geometry and design
   coefficients.
3. Call `analyse(...)` at the design point or at off-design combinations of mass flow and speed.
4. Optionally call `verify_design()`, `print_dimensions()`, `plot_geometry()` or `print_analysis_results(...)`.

#### Literature represented in the implementation

Following literature was used when creating `BarskePump`:

- *The Design of Open Impeller Centrifugal Pumps*, Barske.
- *A Forced Vortex Pump for High Speed, High Pressure, Low Flow Applications*, Lock.
- *Centrifugal Pumps*, Lobanoff et al.
- *Centrifugal Pumps*, Gülich, 4th edition.
- *Rotating And Positive-Displacement For Low-Thrust Rocket Engines Pumps*, Rocketdyne.

The code comments relevant equations, figures, tables and design limits from these works.

#### Design-point quantities

`size_dimensions(...)` takes the fluid, design speed `RPM`, required pressure rise `dp`, mass flow `mdot`, upstream
total pressure `p_upstream` and temperature `T_upstream`. It also allows to choose various design methods for inlet,
impeller-diameter, blade-width, diffuser-outlet and hub sizing.

The inlet density defines the design volumetric flow:

$$
Q_{\mathrm{design}} = \frac{\dot{m}}{\rho(p_{\mathrm{upstream}},T_{\mathrm{upstream}})}.
$$

The required head is not approximated only as $\Delta p/(\rho g)$. `functions.get_H_from_dp(...)` integrates
specific volume, which permits density to vary with pressure:

$$
H_{\mathrm{required}} = \frac{1}{g}\int_{p_{\mathrm{upstream}}}^{p_{\mathrm{upstream}}+\Delta p}
\frac{dp}{\rho(p,T_{\mathrm{upstream}})}.
$$

The design assumes inlet volumetric flow based on upstream density. Static inlet pressure is obtained from the
upstream total pressure with a Bernoulli dynamic-pressure correction. The specific speed is

$$
n_q = \frac{N\sqrt{Q_{\mathrm{design}}}}{H_{\mathrm{required}}^{3/4}},
$$

where $N$ is in rpm.

#### Inlet and passage sizing

The `inlet_sizing_method` selects one of four routes:

- `"Lobanoff"` uses equation 11-22 recalled in the comments,
  $D_0=5.1(Q_{\mathrm{GPM}}/N)^{0.333}$ in inches, and then applies `D_1_over_D_0`.
- `"flow velocity"` uses the specified eye velocity `v_0`:

$$
D_0 = \sqrt{\frac{4Q}{\pi v_0}}.
$$

- `"blade velocity"` uses $D_1=2u_1/\omega$.
- `"flow coefficient"` uses the assigned inlet flow coefficient $\phi_1$:

$$
D_1 = \left(\frac{8Q}{\pi\omega\phi_1}\right)^{1/3}.
$$

Here $\omega=2\pi N/60$. Methods that first determine $D_1$ recover the eye diameter from `D_1_over_D_0`.

$D_2$ is chosen iteratively until requirement the on pump head is met. For every trial outer diameter $D_2$,
the code calculates $u_2=\omega D_2/2$. The diffuser-throat velocity follows from `flow_coefficient_outlet`,
$v_3=\phi_3u_2$, and its diameter follows from continuity. The diffuser outlet is
then sized either with `diffuser_area_ratio` or a prescribed `D_diffuser_outlet`; its length follows from the
diameter change and `diffuser_angle`.

`widths_sizing_method="Gulich"` obtains outlet width from equation 7-1a from Gülich and computes inlet
width from the selected meridional-velocity ratio. `"Rocketdyne"` starts from the modified equation 26 relation

$$
L_1 = \frac{0.25\pi D_1^2}{r\left(\pi D_1-n_b t_1\right)},
$$

where `r_factor` is $r$. `"diameter fraction"` instead sets $L_1=(L_1/D_1)D_1$. In all cases the open flow areas
account for blade blockage:

$$
A_i = L_i\left(\pi D_i-n_b t_i\right).
$$

The inlet and outlet widths are coupled through `V_r_ratio`, and the remaining blade angles, hub dimensions, casing
clearances and optional expeller geometry are derived after the impeller diameter is known.

#### Solving for impeller diameter

`diameter_sizing_method` selects the head model used while sizing $D_2$:

- `"Lobanoff"` evaluates the Lobanoff design-point relation.
- `"Lock"` evaluates the Lock forced-vortex model, including digitised Figure 10a and 10b data from his paper
represented by interpolation functions for $h_0$ and $C_h$.

The inner function `get_impeller_head(D_2)` reconstructs all dimensions that depend on a trial diameter and returns
its predicted design head. SciPy's TOMS 748 root finder solves

$$
H_{\mathrm{required}}-H_{\mathrm{model}}(D_2)=0.
$$

The bracket starts at $1.1D_1$ and extends to twice a diameter estimate based on a typical head coefficient of 1.4.
The converged geometry and both Lock- and Barske-compatible design coefficients are stored in the object. Lobanoff
sizing selects Barske as the default off-design analysis method; Lock sizing selects Lock approach.

#### Hydraulic analysis

`analyse(...)` accepts `fluid`, `mdot`, `RPM`, `p_upstream`, `T_upstream`, `analysis_method`, `K_factor`, `eta_losses`
and `no_prerotation`. It recalculates density, viscosity, flow, static inlet conditions and velocities at the eye,
impeller inlet and outlet, diffuser throat and diffuser outlet for off-design conditions. Optional model coefficients
can be given based on experimental data. If not given, they default to the values stored during sizing.

The analysis routes are:

- `__analysis_Lobanoff(...)`, used for design-point sizing, separates forced-vortex static head and recovered diffuser
  head. With prerotation, the static contribution is $u_2^2/(2g)$; with `no_prerotation=True`, it is
  $(u_2^2-u_1^2)/(2g)$.
- `__analysis_Barske(...)` uses the stored design-point static-head coefficient off design. It converts between static
  and total head with inlet and outlet velocity heads and applies the Barske maximum-throat-velocity cutoff.
- `__analysis_Lock(...)` obtains the ideal Euler head, the operating flow at peak power, the empirical $h_0$ and
  $C_h$ terms, prerotation through `K_factor`, and diffuser loss through `eta_losses`. It also applies Lock's
  vapor-pressure breakdown criterion. 

For all routes, head is converted back to pressure with `functions.get_dP_from_H(...)`. That function numerically
inverts the head integral with the TOMS 748 method. The result dictionary distinguishes static pressure rise `dp`
from total pressure rise `dp_total` and includes the outlet total pressure `p_total_outlet`.

#### Losses, power, temperature and loads

The model evaluates empirical paddle-friction power for the impeller and, when enabled, friction power for an
expeller. Its power balance is

$$
P_{h,\mathrm{useful}}=\dot{m}gH_{\mathrm{total,real}}, \qquad
P_{h,\mathrm{loss}}=\dot{m}gH_{\mathrm{loss}},
$$

$$
P_{\mathrm{total}}=\dot{m}gH_{\mathrm{total,ideal}}+P_{f,\mathrm{impeller}}+P_{f,\mathrm{expeller}}.
$$

Total and static efficiencies divide their respective useful hydraulic powers by `P_total`. Hydraulic and friction
losses heat the fluid according to

$$
T_4=T_{\mathrm{upstream}}+
\frac{P_{h,\mathrm{loss}}+P_{f,\mathrm{total}}}{\dot{m}c_p}.
$$

Torque is `P_total/omega`. Axial load follows the procedure cited from section 9.2 and Tables 9.1-9.3 of Gülich,
with hub, casing-step and optional expeller effects. Radial load follows the procedure cited from Table 9.7.
`verify_design()` checks stored geometry against the Barske and Gülich guidelines described in the comments and
records the pass/fail flags.

The analysis dictionary also exposes head components, coefficients, pressures, velocities, temperature, power
components, efficiencies and axial and radial loads. `print_analysis_results(...)` formats these fields as a table.

### PumpSystem

`PumpSystem` in `DynamicPumps/PumpSystem.py` coordinates one or more hydraulic stages. Construct it with an ordered
list, for example `PumpSystem(stages=[pump])` for one pump or `PumpSystem(stages=[stage_1, stage_2])` for a multistage
assembly.

`analyse(fluid, mdot, RPM, p_upstream, T_upstream)` calls every stage's `analyse(...)` method in list order. All
stages receive the same mass flow and rpm. After a stage is evaluated, its `p_total_outlet` becomes the next stage's
upstream pressure and its `T_4` becomes the next stage's upstream temperature. This is the interface that
future inducer and full-emission pump classes should implement.

The system adds stage static and total heads, static and total pressure rises, useful hydraulic power and shaft
power. It calculates aggregate total and static efficiency from the summed quantities and returns a tuple containing
the system result dictionary and a list of individual stage result dictionaries.

A performance map is generated in this order:

1. Call `sweep_over_RPM_and_mdot(fluid, p_upstream, T_upstream, mdot_range, RPM_range)`.
2. The method creates an rpm-by-mass-flow mesh and calls `analyse(...)` at every grid point, retaining both system
   and per-stage results in `sweep_results`.
3. Call `extract_sweep_results(flattened=False)` to obtain grids of pressure rise, head, efficiencies, power and
   inlet volumetric flow. Use `flattened=True` when one-dimensional arrays are more convenient.
4. Call `plot_pump_map(...)` to plot one curve per rpm and total-efficiency contours. With `QH_map=False`, axes are
   mass flow and static pressure rise; with `QH_map=True`, axes are volumetric flow and static head. Axis limits and
   the contour-count input are accepted by the plotting interface.

The current `DynamicPumps/example.py` follows this exact sequence for a single Barske pump. The same sequence applies
to a multistage list because the map is generated from aggregate system results.

### Fluid

`Fluid` in `DynamicPumps/Fluid.py` provides one property interface to the pump models. It can be constructed in
either of two modes:

- Give `CoolProp_name` to create a CoolProp `AbstractState` using the `TTSE&HEOS` backend. Properties are evaluated
  at the pressure and temperature supplied by the hydraulic analysis.
- Omit `CoolProp_name` and provide constant `density`, `dynamic_viscosity`, `vapor_pressure` and `specific_heat` values.

The constructor always takes a human-readable `name`. The public methods are `get_density(p, T)`,
`get_dynamic_viscosity(p, T)`, `get_kinematic_viscosity(p, T)`, `get_vapor_pressure(T)` and
`get_specific_heat(p, T)`. Kinematic viscosity is calculated from dynamic viscosity and density. Vapor pressure in
CoolProp mode is evaluated on the saturated-vapor line at the requested temperature.

Other classes can use these methods for volumetric flow, head-pressure conversion, Reynolds-dependent friction,
vapor-pressure breakdown checks and loss-induced temperature rise. `PumpSystem` uses density at the system inlet to
report volumetric flow and otherwise passes the same `Fluid` object through every stage.

Constant-property fluids are useful when a compact engineering approximation is adequate or when CoolProp does not
contain the working fluid.
