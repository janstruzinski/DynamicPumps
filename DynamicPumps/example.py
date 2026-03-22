from BarskePump import BarskePump
from Fluid import Fluid

# Create Fluid object
HTP = Fluid(name="HTP", density=1440, dynamic_viscosity=1.25e-3, vapor_pressure=666.61, specific_heat=2619)

# Create impeller geometry
pump = BarskePump()
pump.size_dimensions(fluid=HTP, RPM=3e4, dp=80e5, mdot=1.5, p_upstream=3e5, T_upstream=288.15,
                           inlet_sizing_method="flow velocity", diameter_sizing_method="Lobanoff",
                           widths_sizing_method="diameter fraction", outlet_sizing_method="outlet diameter",
                           hub_sizing_method="diameter fraction", t_hub=3e-3, t_LE=4e-3, t_TE=2e-3, D_inlet=19.05e-3,
                           D_shaft=15e-3, D_hub_over_D_1=1, eta_losses=0.2, diffuser_angle=10,
                           D_diffuser_outlet=19.05e-3, L_1_over_D_1=0.4)
pump.plot_geometry()
