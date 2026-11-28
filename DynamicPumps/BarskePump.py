import warnings
import numpy as np
import functions
import scipy.optimize as opt
import scipy.interpolate as intrp
from tabulate import tabulate
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class BarskePump:
    def __init__(self):
        """A class for sizing and analysis of Barske pump. Barske pump geometry and analysis at design point can be done
         by calling 'size_dimensions' method or 'assign_dimensions' method. Analysis at different operating conditions
         can be performed by calling 'analyse' method."""

        # Create all the class parameters to store pump geometry and analysis results
        # Diameters
        self.D_inlet = None # inlet pipe diameter, m
        self.D_0 = None # impeller eye diameter, m
        self.D_1 = None # impeller inlet diameter, m
        self.D_2 = None # impeller outlet diameter, m
        self.D_3 = None # diffuser throat diameter, m
        self.D_4 = None # diffuser outlet diameter, m
        self.D_hub = None # hub diameter, m
        self.D_1_over_D_0 = None # ratio of D_1 to D_0, -
        self.D_hub_over_D_1 = None # ratio of D_hub to D_1, -
        self.D_shaft = None # shaft diameter, m

        # Widths & lengths
        self.L_1 = None # impeller width (axial length) at the inlet, m
        self.L_2 = None # impeller width at the outlet, m
        self.L_1_over_D_1 = None # ratio of L_1 and D_1, -
        self.L_diffuser = None # diffuser length, m

        # Areas
        self.A_1 = None  # flow area at the inlet of the impeller, m^2
        self.A_2 = None  # flow area at the outlet of the impeller, m^2
        self.A_3 = None # impeller throat diffuser area, m^2
        self.A_4 = None # impeller exit diffuser area, m^2
        self.A_4_over_A_3 = None # diffuser area ratio, -

        # Clearances
        self.s_ax = None # axial clearance, m
        self.s_rad = None  # radial clearance, m
        self.s_rad_hub = None # radial clearance of the hub, m

        # Thicknesses
        self.t_0 = None # impeller hub thickness, m
        self.t_1 = None # impeller blade thickness at inlet, m
        self.t_2 = None # impeller blade thickness at outlet, m

        # Angles
        self.alpha_0 = None # impeller blade forward edge angle wrt. rotation axis, degrees
        self.alpha_1 = None # impeller blade backward edge angle wrt. rotation axis, degrees
        self.alpha_2 = None # sharpening angle of the radial blade wrt. tangent of D_1, degrees
        self.alpha_diffuser = None # diffuser full angle, degrees

        # Splitter blades
        self.splitter_blades = None # boolean whether splitter blades are used
        self.D_splitter = None # splitter blade inlet diameter, m
        self.D_splitter_over_D_1 = None # ratio of D_splitter to D_1, -
        self.t_splitter_1 = None # splitter blade thickness at inlet, m
        self.t_splitter_2 = None # splitter blade thickness at outlet, m

        # Other
        self.n_blades = None # number of main blades, -
        self.V_r_ratio = None # Ratio of outlet to inlet radial (meridional) velocity through the impeller.
        # Also a ratio of A2 to A1.
        self.specific_speed = None # specific speed of the pump at Best Efficiency Point, EU units (m, RPM, m^3/s)
        self.eta_losses_design = None  # fraction of dynamic head lost in the diffuser used for sizing, -.
        self.K_factor_design = None # factor for prerotation at zero flow as a fraction of inlet tip speed used in
        # sizing with Lock's analysis method, -
        self.no_prerotation_design = None # boolean whether no prerotation should be assumed in case of sizing with
        # Lobanoff method.
        self.flow_coefficient_BEP = None # outlet flow coefficient at BEP, -
        self.static_head_coefficient_BEP_Barske = None # static head coefficient at BEP from Lobanoff sizing method for
        # head calculations with Barske method, -.
        self.static_head_coefficient_BEP_Lock = None # static head coefficient at BEP from Lock's method, -.
        # Default analysis method
        self.default_analysis_method = None # default analysis method, which is the same as the one used for sizing.

        # Dictionary to store analysis inputs and results.
        self.analysis_results_design = {"method": None, # Method used for analysis, -
                                        "fluid": None, # Fluid object used for analysis, -
                                        "RPM": None, # Rotations Per Minute, 1 / minute
                                        "omega": None,  # Angular speed, rad / s
                                        "mdot": None, # Massflow through pump, kg / s
                                        "dp": None,  # Static pressure rise across the pump, Pa
                                        "dp_total": None,  # Total pressure rise across the pump, Pa
                                        "Q": None, # Volumetric flow through pump (defined with p_0)
                                        "H_total_real": None, # Real total head of the pump, m
                                        "H_static_real": None,  # Real static head of the pump, m
                                        "H_losses": None,  # Head losses of the pump, m
                                        "H_total_ideal": None,  # Ideal total head of the pump, m
                                        "H_static_ideal": None,  # Ideal static head of the pump, m
                                        "flow_coefficient_inlet": None, # Flow coefficient at the inlet of the impeller, -
                                        "flow_coefficient_outlet": None, # Flow coefficient in the diffuser throat, -
                                        "static_head_coefficient": None, # Static head coefficient of the pump, -
                                        "P_h_useful": None, # Useful pump power, W
                                        "P_h_losses": None, # Hydraulic power losses, W
                                        "P_h_total": None,  # Total hydraulic power, W
                                        "P_total": None,  # Total pump power, W
                                        "P_f_impeller": None,  # Power to overcome friction for the impeller, W
                                        "P_f_total": None, # Total friction power loss, W
                                        "eta_total": None, # Pump total efficiency, -
                                        "eta_static": None,  # Pump total efficiency, -
                                        "eta_losses": None, # Fraction of dynamic head lost in the diffuser, -
                                        "T_upstream": None, # Upstream temperature, K
                                        "rho": None, # Fluid density used for the analysis, kg/m^3
                                        "p_upstream": None,  # Upstream total (tank/reservoir) pressure, Pa
                                        "p_inlet": None, # Pressure in the inlet pipe, Pa
                                        "p_0": None, # Pressure in the impeller eye, Pa
                                        "p_hub": None, # Pressure at the edge of the impeller hub, Pa
                                        "p_2": None, # Pressure at the impeller outlet, Pa
                                        "p_4": None, # Pressure at the diffuser outlet, Pa
                                        "p_total_outlet": None,  # Total pressure at the diffuser outlet, Pa
                                        "p_shaft": None, # Pressure at the impeller shaft, Pa
                                        "v_inlet": None,  # Velocity in the inlet pipe, m/s
                                        "v_0": None, # Velocity at the impeller eye, m/s
                                        "u_1": None, # Blade velocity at the impeller inlet, m/s
                                        "v_1ax": None,  # Absolute flow axial velocity at the impeller inlet, m/s
                                        "v_1m": None,  # Absolute flow meridional velocity at the impeller inlet, m/s
                                        "w_1": None,  # Relative flow velocity at the impeller inlet, m/s
                                        "u_2": None, # Blade velocity at the impeller outlet, m/s
                                        "w_2": None,  # Relative flow velocity at the impeller inlet, m/s
                                        "v_3": None, # Fluid velocity in the diffuser throat, m/s
                                        "v_4": None, # Fluid velocity at the diffuser outlet, m/s
                                        "T_4": None,  # Outlet temperature, K
                                        "F_ax": None, # Axial force, N
                                        "F_rad": None,  # Radial force, N
                                        "Torque": None,  # Torque, N
                                        }

        # Dictionary with flags whether the best practices/guidelines are satisfied
        self.design_checks = {"eye_velocity": None, # Check whether eye velocity guideline is satisfied
                              "LE_velocity": None, # Check whether LE velocity guideline is satisfied
                              "diameter_ratio": None, # Check whether impeller diameters ratio guideline is satisfied
                              "LE_width": None, # Check whether LE width guideline is satisfied
                              "TE_width": None, # Check whether TE width guideline is satisfied
                              "radial_clearance": None, # Check whether radial clearance guideline is satisfied
                              "axial_clearance": None,  # Check whether axial clearance guideline is satisfied
                              "TE_width_clearance_ratio": None, # Check whether TE width to clearance ratio guideline is
                              # satisfied
                              "D_3": None # Check whether throat diameter is smaller than TE width
                              }

        # Constants
        self.g = 9.80665 # gravitational acceleration m/s^2
        self.CMPS_to_GPM = 15850.323140625 # conversion factor between cubic meters per second and gallons per minute
        self.feet_to_m = 0.3048 # conversion factor from feet to meter, ft/m
        self.lb_to_kg = 0.45359237 # conversion factor from pound to kilograms, kg/lb
        self.inch_to_m = 0.0254 # conversion factor from inch to meter, inch/m
        self.HP_to_W = 745.699872  # conversion factor from horsepower to Watt, W/HP

        # Digitized graphs from "A Forced Vortex Pump for High Speed, High Pressure, Low Flow Applications" by Lock.
        # Used for head analysis with his method.
        # First interpolator for h_0 graph for radial blades (upper left most graph from Figure 10a)
        r_ratio = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        n_blades = np.array([2, 4, 8, 16])
        h_0_2_blades = np.array([0.498, 0.482, 0.435, 0.334, 0.190, 0])
        h_0_4_blades = np.array([0.636, 0.636, 0.606, 0.517, 0.332, 0])
        h_0_8_blades = np.array([0.768, 0.768, 0.768, 0.731, 0.518, 0])
        h_0_16_blades = np.array([0.864, 0.864, 0.864, 0.864, 0.753, 0])
        h_0 = np.transpose([h_0_2_blades, h_0_4_blades, h_0_8_blades, h_0_16_blades])
        # 2D interpolator for number of blades <= 16
        h_0_interp_2D = intrp.RegularGridInterpolator(points=(r_ratio, n_blades), values=h_0, method="pchip")
        # 1D interpolator for number of blades > 16
        h_0_interp_16 = intrp.PchipInterpolator(r_ratio, h_0_16_blades)
        # Property to store interpolating function
        self.h_0 = lambda r, n: (h_0_interp_2D([[r, n]])[0] if n <= 16 else h_0_interp_16(r))
        # Now interpolator for C_h graph for radial blades (upper line from Figure 10b)
        blade_spacing_length_ratio = [0.506, 0.606, 0.704, 0.804, 0.904, 1.003, 1.104, 1.202, 1.301, 1.401, 1.501,
                                      1.603, 1.704, 1.803, 1.903, 2.003, 2.104, 2.206, 2.307, 2.406, 2.505]
        C_h = [0.995, 0.994, 0.988, 0.980, 0.967, 0.955, 0.942, 0.926, 0.908, 0.890, 0.872, 0.854, 0.837, 0.819, 0.803,
               0.787, 0.772, 0.757, 0.743, 0.729, 0.717]
        C_h_interp = intrp.interp1d(blade_spacing_length_ratio, C_h, kind="linear", bounds_error=False,
                                    fill_value="extrapolate")
        # Property to store interpolating function
        self.C_h = lambda sigma: np.clip(C_h_interp(sigma), 0.0, 1.0)

    def size_dimensions(self, fluid, RPM, dp, mdot, p_upstream, T_upstream, inlet_sizing_method, diameter_sizing_method,
                        widths_sizing_method, outlet_sizing_method, hub_sizing_method, t_hub, t_LE, t_TE, D_inlet,
                        D_shaft, V_r_ratio=1, n_blades = 5, diffuser_area_ratio = 4, diffuser_angle = 8,
                        flow_coefficient_outlet = 0.8, D_1_over_D_0 = 1.1, D_hub_over_D_1 = 1.1, alpha_1 = 90,
                        s_ax_over_D_2 = 0.01, v_0 = 3.6576, u_1 = 45.72, flow_coefficient_inlet = 0.07,
                        L_1_over_D_1 = 0.25, r_factor = 0.8, eta_losses = 0.194, K_factor = 0.17,
                        no_prerotation = False, D_diffuser_outlet = None, s_rad_hub_over_D_2 = 0.01,
                        splitter_blades = False, D_splitter_over_D_1 = None):
        """A method to size the Barske pump. It updates parameters of the BarskePump object.

        :param Fluid fluid: Fluid object representing fluid used for the sizing of the Barske Pump.
        :param float or int RPM: Design Rotations Per Minute at BEP, 1 / minute
        :param float or int dp: Pressure rise across the pump, Pa
        :param float or int mdot: Massflow through the pump, kg/s
        :param float or int p_upstream: Upstream total (tank/reservoir) pressure, Pa
        :param float or int T_upstream: Upstream temperature, K
        :param string inlet_sizing_method: Method used for sizing of the pump inlet, either "Lobanoff", "flow velocity",
            "blade velocity" or "flow coefficient". If "Lobanoff", eq.11-22 from 'Centrifugal Pumps' by Lobanoff et al.
            will be used for sizing impeller eye diameter D_0. If "flow velocity", D_0 will be sized based on assigned
            velocity v_0. In case of both of these options, impeller inlet diameter D_1 will be calculated from
            D_0 and D_1_over_D_0. If "blade velocity", D_1 is calculated from assigned blade velocity, u_1.
            If "flow coefficient", it is calculated from assigned flow_coefficient_inlet.
            D_0 will be then size from D_1 and D_1_over_D_0.
        :param string diameter_sizing_method: Method used for sizing of the impeller outer diameter D_2.
            It should be either "Lobanoff" or "Lock". If "Lobanoff", eq. 11-4 from 'Centrifugal Pumps'
            by Lobanoff et al. will be used. It is preferable to use equation with diffuser efficiency rather than head
            coefficent, beacause it is more quanitfiable quanity. However, other than that, the analysis is the same as
            proposed by Barkse in "The Design of Open Impeller Centrifugal Pumps". If "Lock", procedure from "A Forced
            Vortex Pump for High Speed, High Pressure, Low Flow Applications" by Lock will be used. This will also set
             the default analysis method for the impeller.
        :param string widths_sizing_method: Method used for sizing of the pump widths (axial lengths), either
            "Gulich", "Rocketdyne" or "diameter fraction". If "Gulich", impeller outlet width L_2 is calculated from
            equation 7-1a from "Centrifugal Pumps" by Gulich. L_1 is then calculated such that constant meridional
            velocity is achieved. If "Rocketdyne", then impeller inlet width L_1 is calculated from eq. 26 from
            "Rotating And Positive-Displacement For Low-Thrust Rocket Engines Pumps" by Rocketdyne.
            If "diameter fraction", L_1 is calculated from assigned L_1_over_D_1. In both of these options, L_2 is then
            calculated from L_1 such that constant meridional velocity is achieved.
        :param string outlet_sizing_method: Method used for sizing the diffuser, either "area ratio" or "outlet diameter".
             If "area ratio", diffuser_area_ratio will be used to calculate outlet diameter. If "outlet diameter", given
             outlet diameter D_4 will be assigned and diffuser_area_ratio will be calculated.
        :param string hub_sizing_method: Method used for sizing the hub diameter, either "diameter fraction" or
            "outlet diameter". If "diameter fraction", D_hub_over_D_1 will be used to calculate hub diameter.
             If "outlet diameter", given outlet diameter D_2 will be assigned and D_hub_over_D_1 will be calculated.
            Hub diameters within a relative tolerance of 1e-12 of D_2 are set exactly to D_2.
        :param float or int t_hub: Hub thickness, m
        :param float or int t_LE: Leading edge (suction side) thickness, m
        :param float or int D_inlet: Inlet pipe diameter, m
        :param float or int D_shaft: Shaft diameter, m
        :param float or int V_r_ratio: Ratio of outlet to inlet radial (meridional) velocites through the impeller, -.
            This is also a ratio of A2 to A1. By default, equal to 1.
        :param float or int t_TE: Trailing edge (suction side) thickness, m
        :param int n_blades: Number of main blades. By default, 5, which is within a range of 3-6 often mentioned for Barske
            impellers.
        :param float or int diffuser_area_ratio: Area ratio of the conical diffuser. Must be greater than 1.
        :param float or int diffuser_angle: Full angle of the conical diffuser in degrees. By default, 8 degrees, which
            is a common value for diffusers.
        :param float or int flow_coefficient_outlet: Flow coefficient of the pump at its outlet, that is, the ratio of
            fluid velocity in the throat v_3 to the outlet blade speed u_2. By default, 0.8.
        :param float or int D_1_over_D_0: Ratio of D_1 (impeller inner diameter) to D_0 (impeller eye diameter).
            By default, 1.1.
        :param float or int D_hub_over_D_1: Ratio of D_hub (hub diameter) to D_1 (impeller inner diameter).
            By default, 1.1. Used in the case hub_sizing_method is "diameter fraction".
        :param float or int alpha_1: impeller blade backward edge angle wrt. rotation axis, degrees. By default,
            90 degrees. Full-diameter hubs are sized with alpha_1 equal to 90 degrees.
        :param float s_ax_over_D_2: Axial clearance s_ax over impeller diameter D_2. By default, 0.01.
            This is a recommended value by Barske in "The Design of Open Impeller Centrifugal Pumps".
        :param float or int v_0: Impeller eye axial velocity, used in "flow velocity" option for inlet_sizing_method.
            By default, 3.6576 m/s. Default value taken from value of 12 ft/s, which is the upper recommended limit by
            Barske in "The Design of Open Impeller Centrifugal Pumps".
        :param float or int u_1: Impeller inlet blade speed, used in "blade velocity" option for inlet_sizing_method.
            By default, 45.72 m/s. Default value taken from value of 150 ft/s, which is the upper recommended limit by
            Barske in "The Design of Open Impeller Centrifugal Pumps".
        :param float or int flow_coefficient_inlet: Flow coefficient at the inlet, used in "flow coefficient" option for
            inlet_sizing_method. It is the ratio of the axial fluid velocity at the impeller inlet v_1ax to the inlet
            blade speed u_1. By default, 0.07, which is the recommended value in "Rotating And Positive-Displacement For
            Low-Thrust Rocket Engines Pumps" by Rocketdyne.
        :param float or int L_1_over_D_1: Ratio of L_1 (axial width at the impeller inlet) to D_1 (impeller inner
            diameter), used in "diameter fraction" option for widths_sizing_method. By default, 0.25. This is the lower
             recommended limit  by Barske in "The Design of Open Impeller Centrifugal Pumps".
        :param float or int r_factor: Semi-empirical factor used in "Rocketdyne" option for widths_sizing_method, which
            is the flow area to meridional area ratio. By default, a value of 0.8 recommended by Rocketdyne.
        :param float or int eta_losses: Fraction of dynamic head lost in the diffuser, -. By default, 0.194. This is the
            value used by Lock in "A Forced Vortex Pump for High Speed, High Pressure, Low Flow Applications"
        :param float or int K_factor: Factor for prerotation at zero flow as a fraction of inlet tip speed, -. Used for
            "Lock" option for diameter_sizing_method. By default, 0.17. Middle value from the range used in
            "A Forced Vortex Pump for High Speed, High Pressure, Low Flow Applications".
        :param bool no_prerotation: Boolean that determines where no prerotation should be assumed for impeller head
            calculations when diameter_sizing_method is "Lobanoff", True or False. By default, False, as recommended in
            Lobanoff.
        :param float s_rad_hub_over_D_2: Ratio of hub radial clearance s_rad_hub to impeller diameter D_2.
            Used for full-diameter hubs. By default, 0.01.
        :param bool splitter_blades: Boolean whether one splitter blade is added halfway between each pair of main
            blades. By default, False. Splitter blades contribute to outlet blockage, but not inlet blockage or
            Lock's blade spacing to length ratio.
        :param float or int D_splitter_over_D_1: Ratio of splitter blade inlet diameter D_splitter to D_1.
            By default, None. Required when splitter_blades is True and ignored otherwise. Must give
            D_1 < D_splitter < D_2. Splitter thickness is linearly interpolated from the main blade at D_splitter
            and equals t_TE at D_2.
        """
        # Fist get fluid density
        rho = fluid.get_density(p_upstream, T_upstream) # kg/s

        # Calculate volumetric flow and the required head. When calculating volumetric flow, it is assumed fluid is
        # incompressible. However, this potential error only has a very minor impact on the sized dimensions, since the
        # impeller is still sized for required head. Therefore, this class can be still used for compressible fluids.
        Q_design = mdot / rho  # m^3 / s
        H_required = functions.get_H_from_dp(fluid, dp, p_upstream, T_upstream) # m

        # Calculate pressure in the inlet pipe. The dynamic head is not large in the inlet, so Bernoulli equation can
        # be used even if fluid is compressible.
        self.D_inlet = D_inlet
        p_static_inlet = p_upstream - 0.5 * rho * (4 * Q_design/(np.pi * self.D_inlet**2))**2

        # Calculate specific speed and assign number of blades
        self.specific_speed = RPM * np.sqrt(Q_design) / (H_required**0.75)
        self.n_blades = n_blades
        self.splitter_blades = splitter_blades
        # Splitter blades only contribute to the blade count at the impeller outlet
        n_blades_outlet = self.n_blades * (2 if self.splitter_blades else 1)
        if self.splitter_blades and D_splitter_over_D_1 is None:
            raise ValueError("D_splitter_over_D_1 cannot be None if splitter_blades is True.")

        # Calculate angular speed of rotation
        omega = RPM * 2 * np.pi / 60 # rad / s

        # Size the inlet diameters
        self.D_1_over_D_0 = D_1_over_D_0
        # If inlet sizing method is 'Lobanoff' or 'flow velocity', first size the eye diameter and then impeller inlet
        # diameter
        if inlet_sizing_method in ("Lobanoff", "flow velocity"):
            if inlet_sizing_method == "Lobanoff":
                self.D_0 = (5.1 * (Q_design * self.CMPS_to_GPM / RPM)**0.333) * self.inch_to_m # m
            elif inlet_sizing_method == "flow velocity":
                self.D_0 = np.sqrt(4 * Q_design / (np.pi * v_0))    # m
            self.D_1 = self.D_0 * self.D_1_over_D_0 # m
        # If inlet sizing method is inlet flow coefficient or inlet blade speed, first size the impeller inlet diameter
        # and then the eye diameter
        elif inlet_sizing_method in ("blade velocity", "flow coefficient"):
            if inlet_sizing_method == "blade velocity":
                self.D_1 = 2 * u_1 / omega  # m
            elif inlet_sizing_method == "flow coefficient":
                self.D_1 = (8 * Q_design / (np.pi * omega * flow_coefficient_inlet))**(1/3) # m
            self.D_0 = self.D_1 / self.D_1_over_D_0 # m
        # If wrong option was assigned, raise error
        else:
            raise ValueError("inlet_sizing_method must be 'Lobanoff', 'flow velocity', 'blade velocity' or"
                             " 'flow coefficient'")

        # Check the hub sizing method before solving for the outlet diameter
        if hub_sizing_method not in ("diameter fraction", "outlet diameter"):
            raise ValueError("hub_sizing_method must be 'diameter fraction' or 'outlet diameter'")

        # If alpha_1 is not 90 degrees and D_hub_over_D_1 is different from 1 or outlet diameter sizing method is
        # used, alpha_1 will be changed to 90 degrees.
        if alpha_1 != 90 and (hub_sizing_method == "outlet diameter" or D_hub_over_D_1 != 1):
            warnings.warn("Alpha_1 is not 90 degrees, but D_hub_over_D_1 is not 1."
                          " To ensure feasible geometry, alpha_1 is set to 90 degrees.")
            alpha_1 = 90 # deg

        # Assign thicknesses and other known quantities
        self.t_0 = t_hub # m
        self.t_1 = t_LE # m
        self.t_2 = t_TE # m
        self.alpha_1 = alpha_1 # degrees
        self.alpha_diffuser = diffuser_angle # degrees
        self.eta_losses_design = eta_losses # -
        self.K_factor_design = K_factor # -
        self.flow_coefficient_BEP = flow_coefficient_outlet # -
        self.no_prerotation_design = no_prerotation # -
        self.A_4_over_A_3 = diffuser_area_ratio # -
        self.D_shaft = D_shaft # m
        self.V_r_ratio = V_r_ratio # -

        # Find outlet diameter that satisfies requirementsa
        # First define a function that calculates impeller's dimensions and H as a function of D_2
        def get_impeller_head(D_2):
            # Set class property
            self.D_2 = D_2
            # Calculate blade tip speed
            u_2 = D_2 * omega / 2 # m/s

            # Size the diffuser from the assigned outlet flow coefficient
            v_3 = u_2 * self.flow_coefficient_BEP # m/s
            self.D_3 = np.sqrt(4 * Q_design / (np.pi * v_3)) # m
            self.A_3 = np.pi * (self.D_3 / 2)**2 # m^2
            # If diffuser sizing is done through area ratio
            if outlet_sizing_method == "area ratio":
                self.A_4 = self.A_4_over_A_3 * self.A_3  # m^2
                self.D_4 = 2 * np.sqrt(self.A_4 / np.pi)  # m
            # If it is done with outlet diameter
            elif outlet_sizing_method == "outlet diameter":
                if D_diffuser_outlet is None:
                    raise ValueError("D_diffuser_outlet cannot be None if outlet_sizing_method is 'outlet diameter'.")
                self.D_4 = D_diffuser_outlet # m
                self.A_4 = np.pi * (self.D_4 / 2)**2 # m2
            # If wrong input was given
            else:
                raise ValueError("outlet_sizing_method must be 'area ratio' or 'outlet diameter'.")
            diffuser_area_ratio = self.A_4 / self.A_3
            L_diffuser = ((self.D_4 - self.D_3) / 2) / np.tan((self.alpha_diffuser/2) * np.pi / 180) # m

            # If blade width sizing method is Gulich, first get L_2 and then get L_1
            if widths_sizing_method == "Gulich":
                self.L_2 = (0.02 + 0.5 * (self.specific_speed/100) - 0.03 * (self.specific_speed/100)**2
                       - 0.04 * (self.specific_speed/100)**3) * D_2 # m
                # Calculate L_1, while taking blade thickness into account
                self.A_2 = D_2 * self.L_2 * np.pi - n_blades_outlet * self.L_2 * self.t_2 # m^2
                self.L_1 = self.A_2 * self.V_r_ratio / (self.D_1 * np.pi - self.n_blades * self.t_1)  # m
                self.A_1 = self.D_1 * self.L_1 * np.pi - self.n_blades * self.L_1 * self.t_1 # m^2
            # If it is Barske or Rocketdyne method, first get L_1 and then get L_2
            elif widths_sizing_method in ("diameter fraction", "Rocketdyne"):
                if widths_sizing_method == "diameter fraction":
                    self.L_1 = L_1_over_D_1 * self.D_1   # m
                elif widths_sizing_method == "Rocketdyne":
                    # Rocketdyne method was slightly changed so that blade thickness is taken into account
                    self.L_1 = ((0.25 * np.pi * self.D_1**2) / (self.D_1 * np.pi - self.n_blades * self.t_1)) \
                               / r_factor # m
                # Calculate L_2, while taking blade thickness into account
                self.A_1 = self.D_1 * self.L_1 * np.pi - self.n_blades * self.L_1 * self.t_1 # m^2
                self.L_2 = self.A_1 / ((D_2 * np.pi - n_blades_outlet * self.t_2) * self.V_r_ratio)  # m
                self.A_2 = D_2 * self.L_2 * np.pi - n_blades_outlet * self.L_2 * self.t_2 # m^2
            else:
                raise ValueError("widths_sizing_method must be 'Gulich', 'diameter fraction' or 'Rocketdyne'")

            # Calculate the sharpening angle of the suction side wrt. tangent of D_1. It is the same as local flow
            # angle. It is assumed that the fluid flows through whole perimeter of D_1.
            alpha_2 = np.arctan((Q_design / self.A_1) / u_1) * 180 / np.pi # degrees

            # Calculate axial and radial clearances between the casing and the impeller.
            s_ax = s_ax_over_D_2 * D_2 # m
            # Radial clearance is calculated such that centerline of the diffuser throat is at D_2
            s_rad = self.D_3 / 2 # m

            # Analyse performance either with Lobanoff or Lock method
            if diameter_sizing_method == "Lock":
                H = self.__analysis_Lock(u_2, u_1, Q_design, fluid, p_static_inlet, T_upstream, self.eta_losses_design,
                                         self.K_factor_design)[0]
            elif diameter_sizing_method == "Lobanoff":
                H = self.__analysis_Lobanoff(u_1, u_2, Q_design)[0]
            else:
                raise ValueError("diameter_sizing_method must be 'Lock' or 'Lobanoff'")

            return (H, u_2, self.D_3, self.D_4, self.A_1, self.A_2, self.A_3, self.A_4, diffuser_area_ratio, L_diffuser,
                    self.L_1, self.L_2, alpha_2, s_ax, s_rad)

        # Now solve the function for D_2 and get all other remaining dimensions. Typical head coefficient for Barske
        # impellers is about 1.4 (Gulich). Bisection is used, so the diameter range is from 1.1x inner diameter to
        # 2x diameter sized with head coefficient of 1.4.
        D_2_estimate = (2 / omega) * np.sqrt(2 * self.g * H_required / 1.4)
        self.D_2 = opt.toms748(f=lambda x: H_required - get_impeller_head(x)[0], a=self.D_1 * 1.1, b=2*D_2_estimate)
        # Now get the remaining results
        (H_design, u_2, self.D_3, self.D_4, self.A_1, self.A_2, self.A_3, self.A_4, self.A_4_over_A_3, self.L_diffuser,
         self.L_1, self.L_2, self.alpha_2, self.s_ax, self.s_rad) = get_impeller_head(self.D_2)

        # Set the hub diameter using the converged outlet diameter
        if hub_sizing_method == "diameter fraction":
            self.D_hub = self.__set_hub_diameter(D_hub_over_D_1 * self.D_1, self.D_2) # m
        else:
            self.D_hub = self.D_2 # m
        self.D_hub_over_D_1 = self.D_hub / self.D_1 # -

        # Set the hub clearance and trailing-edge position from the resulting geometry
        if self.D_hub == self.D_2:
            if s_rad_hub_over_D_2 is None:
                raise ValueError("s_rad_hub_over_D_2 must be provided for a full-diameter hub.")
            self.s_rad_hub = s_rad_hub_over_D_2 * self.D_2 # m
            dy_TE = self.t_0 # m
        else:
            self.s_rad_hub = None
            dy_TE = np.tan((90 - self.alpha_1) * np.pi / 180) * ((self.D_2 - self.D_1) / 2) # m

        # Calculate the angle of the forward edge from the axial and radial distances
        self.alpha_0 = np.arctan(((self.D_2 - self.D_1) / 2) / (self.L_1 + self.t_0 - self.L_2 - dy_TE)) \
                       * 180 / np.pi # degrees

        # Size the splitter blades after the outlet diameter has converged. If not used, their geometry is None.
        if not self.splitter_blades:
            self.D_splitter = None
            self.D_splitter_over_D_1 = None
            self.t_splitter_1 = None
            self.t_splitter_2 = None
        else:
            self.D_splitter = D_splitter_over_D_1 * self.D_1 # m
            if not self.D_1 < self.D_splitter < self.D_2:
                raise ValueError("D_splitter_over_D_1 must give D_1 < D_splitter < D_2.")
            self.D_splitter_over_D_1 = D_splitter_over_D_1 # -
            self.t_splitter_1 = self.t_1 + (self.t_2 - self.t_1) * (self.D_splitter - self.D_1) \
                                / (self.D_2 - self.D_1) # m
            self.t_splitter_2 = self.t_2 # m

        # Calculate ratio of L_1 to D_1
        self.L_1_over_D_1 = self.L_1 / self.D_1

        # Regardless with what method diameter was sized, obtain static head coefficient for Lobanoff method.
        # This is necessary in case the sized geometry is analyzed in the future with methods proposed by Barske in
        # "The Design of Open Impeller Centrifugal Pumps". The head coefficient follows the convention in
        # "Centrifugal Pumps" by Gulich.
        self.static_head_coefficient_BEP_Barske =\
            self.__analysis_Lobanoff(u_1, u_2, Q_design)[2]

        # For informational purposes, obtain static head coefficient for Lock's method as well
        self.static_head_coefficient_BEP_Lock =\
            self.__analysis_Lock(u_2, u_1, Q_design, fluid, p_static_inlet, T_upstream, self.eta_losses_design,
                                 self.K_factor_design)[5]

        # With all dimensions known, full analysis can be performed to get more data about flow and performance.
        # At the same time, default analysis method can be set.
        if diameter_sizing_method == "Lobanoff":
            self.default_analysis_method = "Barske"
        else:
            self.default_analysis_method = "Lock"
        self.analysis_results_design = self.analyse(fluid, mdot, RPM, p_upstream, T_upstream,
                                                    self.default_analysis_method, K_factor, eta_losses, no_prerotation)

        # Print geometry
        self.print_dimensions()
        # Print analysis results
        self.print_analysis_results(self.analysis_results_design)
        # Finally, design can be verified
        self.verify_design()

    def assign_dimensions(self, fluid, RPM, Q_start, p_upstream, T_upstream, D_inlet, D_0, D_1, D_2, D_3, D_4,
                          D_hub, D_shaft, L_1, L_2, s_ax, s_rad, t_hub, t_LE, t_TE, alpha_0, alpha_1, alpha_2,
                          n_blades, diffuser_angle, eta_losses=0.194, K_factor=0.17, s_rad_hub=None,
                          splitter_blades=False, D_splitter=None, t_splitter_1=None, t_splitter_2=None):
        """A method to manually assign the Barske pump geometry. It updates parameters of the BarskePump object and
        finds the Best Efficiency Point using Lock's analysis method.

        :param Fluid fluid: Fluid object representing fluid used for the analysis of the Barske Pump.
        :param float or int RPM: Design Rotations Per Minute at BEP, 1 / minute.
        :param float or int Q_start: Starting volumetric flow for the BEP solution, m^3 / s.
        :param float or int p_upstream: Upstream total (tank/reservoir) pressure, Pa.
        :param float or int T_upstream: Upstream temperature, K.
        :param float or int D_inlet: Inlet pipe diameter, m.
        :param float or int D_0: Impeller eye diameter, m.
        :param float or int D_1: Impeller inlet diameter, m.
        :param float or int D_2: Impeller outlet diameter, m.
        :param float or int D_3: Diffuser throat diameter, m.
        :param float or int D_4: Diffuser outlet diameter, m.
        :param float or int D_hub: Hub diameter, m. Diameters within a relative tolerance of 1e-12 of D_2 are set to D_2.
        :param float or int D_shaft: Shaft diameter, m.
        :param float or int L_1: Impeller width (axial length) at the inlet, m.
        :param float or int L_2: Impeller width at the outlet, m.
        :param float or int s_ax: Axial clearance, m.
        :param float or int s_rad: Radial clearance, m.
        :param float or int t_hub: Hub thickness, m.
        :param float or int t_LE: Leading edge (suction side) thickness, m.
        :param float or int t_TE: Trailing edge (suction side) thickness, m.
        :param float or int alpha_0: Impeller blade forward edge angle wrt. rotation axis, degrees.
        :param float or int alpha_1: Impeller blade backward edge angle wrt. rotation axis, degrees.
            Must be 90 degrees for a full-diameter hub.
        :param float or int alpha_2: Sharpening angle of the radial blade wrt. tangent of D_1, degrees.
        :param int n_blades: Number of main blades.
        :param float or int diffuser_angle: Full angle of the conical diffuser, degrees.
        :param float or int eta_losses: Fraction of dynamic head lost in the diffuser, -. By default, 0.194.
        :param float or int K_factor: Factor for prerotation at zero flow as a fraction of inlet tip speed, -.
            By default, 0.17.
        :param float or int s_rad_hub: Radial clearance of the hub, m. By default, None. Required when D_hub is equal
            to D_2 and ignored otherwise.
        :param bool splitter_blades: Boolean whether one splitter blade is added halfway between each pair of main
            blades. By default, False.
        :param float or int D_splitter: Splitter blade inlet diameter, m. By default, None. Required when
            splitter_blades is True and ignored otherwise. Must give D_1 < D_splitter < D_2.
        :param float or int t_splitter_1: Splitter blade thickness at its inlet, m. By default, None. Required when
            splitter_blades is True and ignored otherwise.
        :param float or int t_splitter_2: Splitter blade thickness at its outlet, m. By default, None. Required when
            splitter_blades is True and ignored otherwise.
        """

        # Check if the hub equals impeller outer diameter
        D_hub = self.__set_hub_diameter(D_hub, D_2) # m
        if D_hub == D_2:
            # If it does, alpha_1 must be 90 degrees, since hub is flat
            if alpha_1 != 90:
                raise ValueError("alpha_1 must be 90 degrees for a full-diameter hub (D_hub = D_2).")
            # Also hub radial clearance must be provided
            if s_rad_hub is None:
                raise ValueError("s_rad_hub must be provided for a full-diameter hub.")

        # Assign required geometry
        self.D_inlet = D_inlet # m
        self.D_0 = D_0 # m
        self.D_1 = D_1 # m
        self.D_2 = D_2 # m
        self.D_3 = D_3 # m
        self.D_4 = D_4 # m
        self.D_hub = D_hub # m
        self.D_shaft = D_shaft # m
        self.L_1 = L_1 # m
        self.L_2 = L_2 # m
        self.s_ax = s_ax # m
        self.s_rad = s_rad # m
        self.t_0 = t_hub # m
        self.t_1 = t_LE # m
        self.t_2 = t_TE # m
        self.alpha_0 = alpha_0 # degrees
        self.alpha_1 = alpha_1 # degrees
        self.alpha_2 = alpha_2 # degrees
        self.n_blades = n_blades
        self.alpha_diffuser = diffuser_angle # degrees

        # Calculate diffuser geometry and geometric ratios
        self.A_3 = np.pi * (self.D_3 / 2)**2 # m^2
        self.A_4 = np.pi * (self.D_4 / 2)**2 # m^2
        self.A_4_over_A_3 = self.A_4 / self.A_3 # -
        self.L_diffuser = ((self.D_4 - self.D_3) / 2) \
                          / np.tan((self.alpha_diffuser / 2) * np.pi / 180) # m
        self.D_1_over_D_0 = self.D_1 / self.D_0 # -
        self.D_hub_over_D_1 = self.D_hub / self.D_1 # -
        self.L_1_over_D_1 = self.L_1 / self.D_1 # -

        # Assign the radial clearance for a full-diameter hub
        self.s_rad_hub = s_rad_hub if D_hub == D_2 else None # m

        # Assign optional splitter blade geometry
        self.splitter_blades = splitter_blades
        if not self.splitter_blades:
            self.D_splitter = None
            self.D_splitter_over_D_1 = None
            self.t_splitter_1 = None
            self.t_splitter_2 = None
        else:
            if D_splitter is None or t_splitter_1 is None or t_splitter_2 is None:
                raise ValueError("D_splitter, t_splitter_1 and t_splitter_2 cannot be None if splitter_blades is True.")
            if not self.D_1 < D_splitter < self.D_2:
                raise ValueError("D_splitter must give D_1 < D_splitter < D_2.")
            self.D_splitter = D_splitter # m
            self.D_splitter_over_D_1 = self.D_splitter / self.D_1 # -
            self.t_splitter_1 = t_splitter_1 # m
            self.t_splitter_2 = t_splitter_2 # m

        # Calculate impeller flow areas including blade blockage and meridional velocity ratio
        self.A_1 = self.L_1 * (self.D_1 * np.pi - self.n_blades * self.t_1) # m^2
        outlet_blockage = self.n_blades * self.t_2
        if self.splitter_blades:
            outlet_blockage += self.n_blades * self.t_splitter_2
        self.A_2 = self.L_2 * (self.D_2 * np.pi - outlet_blockage) # m^2
        self.V_r_ratio = self.A_1 / self.A_2 # -

        # Assign design analysis inputs. Other design properties are obtained after finding BEP.
        self.eta_losses_design = eta_losses # -
        self.K_factor_design = K_factor # -
        self.default_analysis_method = "Lock"
        self.no_prerotation_design = False  # Use the prerotation assumption for the Barske head coefficient, when it
        # is calculated after analysing the impeller with Lock's method

        # It is desirable to find Best Efficiency Point (BEP) to determine stage coefficients at design point.
        # Calculate quantities which remain constant while searching for BEP
        rho = fluid.get_density(p_upstream, T_upstream) # kg/m^3
        kinematic_viscosity = fluid.get_kinematic_viscosity(p_upstream, T_upstream) # m^2 / s
        omega = RPM * 2 * np.pi / 60 # rad / s
        u_1 = omega * self.D_1 / 2 # m/s
        u_2 = omega * self.D_2 / 2 # m/s
        P_f_impeller = self.__calculate_friction_power(rho, kinematic_viscosity, RPM) # W

        # Define function that calculates static efficiency for iterative BEP search
        def get_static_efficiency(Q):
            v_inlet = 4 * Q / (np.pi * self.D_inlet**2) # m/s
            p_inlet = p_upstream - 0.5 * rho * v_inlet**2 # Pa
            H_static_real, _, _, H_total_ideal, _, _, _, _ = \
                self.__analysis_Lock(u_2, u_1, Q, fluid, p_inlet, T_upstream, self.eta_losses_design,
                                     self.K_factor_design)
            mdot = rho * Q # kg/s
            P_total = mdot * H_total_ideal * self.g + P_f_impeller # W
            eta_static = mdot * H_static_real * self.g / P_total # -
            return eta_static

        # Ensure initial Q given is not zero
        if Q_start <= 0:
            raise ValueError("Q_start must be greater than zero.")

        # Starting from the user-given estimate, find an upper bound of the columetric flow which encloses the
        # efficiency maximum. This will allow to use bracketing scheme for finding BEP volumetric flow later on.
        Q_upper = Q_start # m^3 / s
        eta_upper = get_static_efficiency(Q_upper)
        # In case Q_start is beyond head breakdown, halve it until positive efficiency is reached
        for i in range(100):
            if eta_upper > 0:
                break
            Q_upper /= 2
            eta_upper = get_static_efficiency(Q_upper)
        else:
            raise RuntimeError("Could not find positive static efficiency starting from Q_start.")
        # In case Q_upper is below BEP volumetric flow value, double Q_upper until it is above.
        # First check if efficiency at halved flow is higher.
        eta_lower = get_static_efficiency(Q_upper / 2)
        # If it is smaller or equal, it means Q_upper must be increased.
        if eta_lower <= eta_upper:
            # Double the upper flow until efficiency stops increasing, placing the maximum inside the bracket
            for i in range(100):
                Q_trial = 2 * Q_upper # m^3 / s
                eta_trial = get_static_efficiency(Q_trial)
                Q_upper = Q_trial
                if eta_trial <= eta_upper:
                    break
                eta_upper = eta_trial
            else:
                raise RuntimeError("Could not bracket the maximum static efficiency.")

        # Now the BEP flow should be somewhere between 0 and Q_upper.
        # Bracketing scheme can be used to efficiently find the exact value.
        solution = opt.minimize_scalar(fun=lambda Q: -get_static_efficiency(Q), bounds=(0, Q_upper), method="bounded",
                                       options={"xatol": max(Q_upper * 1e-8, 1e-12)})
        if not solution.success:
            raise RuntimeError("BEP volumetric flow solution did not converge.")
        Q_design = solution.x # m^3 / s

        # Calculate design coefficients and perform full analysis at BEP
        v_inlet = 4 * Q_design / (np.pi * self.D_inlet**2) # m/s
        p_inlet = p_upstream - 0.5 * rho * v_inlet**2 # Pa
        Lock_results = self.__analysis_Lock(u_2, u_1, Q_design, fluid, p_inlet, T_upstream,
                                            self.eta_losses_design, self.K_factor_design)
        self.flow_coefficient_BEP = 4 * Q_design / (np.pi * self.D_3**2 * u_2) # -
        self.static_head_coefficient_BEP_Barske = self.__analysis_Lobanoff(u_1, u_2, Q_design)[2]
        self.static_head_coefficient_BEP_Lock = Lock_results[5]
        mdot_design = rho * Q_design # kg/s
        self.analysis_results_design = self.analyse(fluid, mdot_design, RPM, p_upstream, T_upstream,
                                                    self.default_analysis_method, K_factor, eta_losses)
        H_design = self.analysis_results_design["H_static_real"] # m
        self.specific_speed = RPM * np.sqrt(Q_design) / (H_design**0.75)

        # Print geometry
        self.print_dimensions()
        # Print analysis results
        self.print_analysis_results(self.analysis_results_design)
        # Finally, design can be verified
        self.verify_design()

    @staticmethod
    def __set_hub_diameter(D_hub, D_2):
        """Set a full-diameter hub exactly to D_2 if it is within numerical tolerance or reject an oversized hub."""
        # Treat differences from floating-point rounding as equality. In this case, hub diameter is set to outlet
        # diameter.
        if np.isclose(D_hub, D_2, rtol=1e-12, atol=0):
            return D_2
        # If hub diameter is smaller than outlet diameter, nothing is changed.
        if D_hub < D_2:
            return D_hub
        # If it is above the outlet diameter, an error is raised.
        raise ValueError(f"Hub diameter D_hub ({D_hub:.16g} m) must be at most the impeller outlet diameter "
                         f"D_2 ({D_2:.16g} m).")

    def __calculate_friction_power(self, rho, kinematic_viscosity, RPM):
        """A method to calculate paddle power required to overcome friction for the impeller.

        :param float or int rho: Fluid density, kg / m^3.
        :param float or int kinematic_viscosity: Fluid kinematic viscosity, m^2 / s.
        :param float or int RPM: Rotations Per Minute, 1 / minute.

        :return: Paddle power required to overcome friction for the impeller, W.
        :rtype: float
        """

        dummy_1a = 0.6e-6 * rho / self.lb_to_kg * self.feet_to_m**3
        dummy_1b = (kinematic_viscosity / self.feet_to_m**2)**0.2 * (RPM / 1000)**2.8
        dummy_1c = np.sin(self.alpha_0 * np.pi / 180)**(-1) + np.sin(self.alpha_1 * np.pi / 180)**(-1)
        dummy_1d = (self.D_2 / self.inch_to_m)**4.6
        dummy_1e = 9.2 * (self.D_1 / self.inch_to_m)**3.6 * self.L_1 / self.inch_to_m
        P_f_impeller = dummy_1a * dummy_1b * (dummy_1c * dummy_1d + dummy_1e) * self.HP_to_W # W
        return P_f_impeller

    def __analysis_Lock(self, u_2, u_1, Q, fluid, p_inlet, T_inlet, eta_losses, K_factor):
        """A method to analyse the pump using Lock's method presented in 'A Forced
            Vortex Pump for High Speed, High Pressure, Low Flow Applications'.

        :param float or int u_2: Tip velocity of the impeller, m/s
        :param float or int u_1: Leading edge velocity of the impeller, m/s
        :param float or int Q: Volumetric flow, m^3 / s.
        :param Fluid fluid: Object representing fluid used for the analysis.
        :param float or int p_inlet: Static pressure in the inlet pipe, Pa.
        :param float or int T_inlet: Static temperature in the inlet pipe, K.
        :param float or int eta_losses: Fraction of dynamic head lost in the diffuser, -.
        :param float or int K_factor: Factor for prerotation at zero flow as a fraction of inlet tip speed, -.

        :return: Tuple containing the results of the Barske pump analysis using Lock's method:
            - H_static_real (float): Real static head.
            - H_total_real (float): Real total head.
            - H_loss (float): Total head loss.
            - H_total_ideal (float): Ideal total head.
            - H_static_ideal (float): Ideal static head.
            - head_coefficient_static (float): Static head coefficient.
            - head_coefficient_total (float): Total head coefficient.
            - H_s (float): Static head at outlet.
        :rtype: tuple[float, float, float, float, float, float, float, float]
        """

        # First find h_0 and C_h factors from digitalized Figure 10a and Figure 10b from Lock. To do so, ratio of
        # impeller radii and blade spacing to length ratio need to be found. Only main blades are counted here.
        r_ratio = self.D_1 / self.D_2
        blade_spacing_length_ratio = (self.D_1 * np.pi / self.n_blades) / ((self.D_2 - self.D_1) / 2)
        # Use interpolators created during object construction
        h_0 = self.h_0(r_ratio, self.n_blades)
        C_h = self.C_h(blade_spacing_length_ratio)

        # Now determine Q_ops for which peak power value occurs. Lock derived the expression for imperial units and some
        # simplifications, but here derived solution for SI units is used.
        dummy_1a = (h_0 * u_2**2) / self.g
        dummy_1b = eta_losses * 24 * (self.g * self.D_3**4 * np.pi**2)**(-1) * (1 - (self.D_3 / self.D_4)**2)**2
        dummy_1c = 24 * (self.D_4**(-4) - self.D_inlet**(-4)) / (np.pi**2 * self.g)
        Q_ops = np.sqrt(dummy_1a / (dummy_1b + dummy_1c)) # m^3 / s

        # First calculate total ideal head. It is assumed equation for total head without prerotation is appropriate,
        # since this is the formula recalled by Lock. Futhermore, K_factors are small and deacrease with flowrate,
        # which supports using the formula below.
        H_total_ideal = (2 * u_2**2 - u_1**2) / (2 * self.g)
        # Then calculate total static head
        H_static_ideal = H_total_ideal + (- dummy_1c / 3) * Q**2
        # Now calculate total head without diffuser losses. Lock's method was modified for the case in which
        # meridional velocity is not constant, hence the term with K coefficient is also multiplied by V_r ratio.
        H_total_wo_diff_loss = dummy_1a - C_h * K_factor * self.V_r_ratio * (u_2**2 / self.g) * \
                               (self.D_1 / self.D_2) * (1 - (Q / Q_ops))
        # Now calculate static head without diffuser losses
        H_static_wo_diff_loss = H_total_wo_diff_loss + (- dummy_1c / 3) * Q**2
        # Calculate diffuser losses
        H_loss_diff = (dummy_1b / 3) * Q**2
        # Then calculate real total head
        H_total_real = H_total_wo_diff_loss - H_loss_diff
        # Now calculate real static head
        H_static_real = H_static_wo_diff_loss - H_loss_diff
        # Calculate total losses
        H_loss = H_total_ideal - H_total_real
        # Also calculate static head due to forced vortex. This is total head minus impeller exit velocity.
        # This expression was derived from first principles, since Lock's does not separate total head into static and
        # dynamic ones like Barske. Hence, it is recommended to use __analysis_Barske to get H_s.
        H_s = H_total_wo_diff_loss - u_2**2 / (2 * self.g)

        # Also determine whether Q is above the maximum volumetric flow. First determine head at vapor pressure wrt.
        # inlet pressure.
        H_vp = functions.get_H_from_dp(fluid=fluid, pressure_rise=fluid.get_vapor_pressure(T_inlet) - p_inlet,
                                       p_0=p_inlet, T_0=T_inlet)
        # Now determine the head in the throat wrt. inlet pressure
        H_3 = H_static_real + H_loss_diff - 8 * Q**2 * (self.D_3**(-4) - self.D_4**(-4)) / (self.g * np.pi**2)
        # If that head is equal or smaller than H_3, the H-Q curve breaks down according to Lock. It must be remarked
        # that Q_ops may be greater than the limiting Q value.
        if H_3 <= H_vp:
            # H_loss becomes the total ideal head
            H_loss = H_total_ideal
            # Total real head becomes zero
            H_total_real = 0
            # Static head is then just the difference in dynamic heads
            H_static_real = (- dummy_1c / 3) * (Q**2)

        # Calculate head coefficients (as defined in Gulich)
        head_coefficient_static = 2 * self.g * H_static_real / u_2**2
        head_coefficient_total = 2 * self.g * H_total_real / u_2**2

        # Now return all the results
        return (H_static_real, H_total_real, H_loss, H_total_ideal, H_static_ideal, head_coefficient_static,
                head_coefficient_total, H_s)

    def __analysis_Lobanoff(self, u_1, u_2, Q):
        """A method to analyse the pump at design point using eq.11-22 from 'Centrifugal Pumps' by Lobanoff et al.

        :param float or int u_1: Velocity of the leading edge of the impeller, m/s
        :param float or int u_2: Tip velocity of the impeller, m/s
        :param float or int Q: Volumetric flow, m^3 / s.

        :return: Tuple containing the results of the Barske pump analysis at design point using Lobanoff's method:
        - H_static_real (float): Real static head.
        - H_total_real (float): Real total head.
        - head_coefficient_static (float): Static head coefficient.
        - head_coefficient_total (float): Total head coefficient.
        :rtype: tuple[float, float, float, float]
        """

        # Calculate static head due to forced vortex. First if it is assumed there is no pre-rotation.
        if self.no_prerotation_design:
            Hs = (u_2**2 - u_1**2) / (2 * self.g)
        # Then if it is assumed there is a prerotation
        else:
            Hs = u_2**2 / (2 * self.g)
        # Now calculate the static recovery from the dynamic head
        Hd = (1 - self.eta_losses_design) * (u_2 * self.flow_coefficient_BEP)**2 / (2 * self.g) * \
             (1 - (self.A_3 / self.A_4)**2)
        # Now calculate the real total head. Hd is just a static recovery from dynamic head of the impeller, so dynamic
        # head at the diffuser outlet must be still added
        v_4 = 4 * Q / (np.pi * self.D_4**2)
        H_total_real = Hs + Hd + v_4**2 / (2 * self.g)
        # Finally calculate the real static head
        v_inlet = 4 * Q / (np.pi * self.D_inlet**2)
        H_static_real = H_total_real + (v_inlet**2 - v_4**2) / (2 * self.g)
        # Calculate the total and static head coefficients (as defined in Gulich)
        head_coefficient_static = 2 * self.g * H_static_real / u_2**2
        head_coefficient_total = 2 * self.g * H_total_real / u_2**2
        return H_static_real, H_total_real, head_coefficient_static, head_coefficient_total
    
    def __analysis_Barske(self, u_1, u_2, v_inlet, v_3, v_4, no_prerotation):
        """A method to analyse the pump using traditional Barske approach with head coefficient. Unlike
         method analysis_Lobanoff, can be used off design point. It is asummed in this approach that head coefficient
         is constant.

        :param float or int u_1: Velocity of the leading edge of the impeller, m/s
        :param float or int u_2: Tip velocity of the impeller, m/s
        :param float or int v_inlet: Velocity in the inlet pipe, m / s
        :param float or int v_3: Velocity in the diffuser throat, m / s
        :param float or int v_4: Velocity in the diffuser outlet, m / s
        :param bool no_prerotation: Boolean whether no prerotation assumption should be used. Either True or False.

        :return: Tuple containing the results of the Barske pump analysis using method with head coefficient:
            - H_static_real (float): Real static head.
            - H_total_real (float): Real total head.
            - H_loss (float): Total head loss.
            - H_total_ideal (float): Ideal total head.
            - H_static_ideal (float): Ideal static head.
            - head_coefficient_static (float): Static head coefficient.
            - head_coefficient_total (float): Total head coefficient.
            - H_s (float): Static head at outlet.
        :rtype: tuple[float, float, float, float, float, float, float, float]
        """

        # Calculate ideal static head due to forced vortex
        if no_prerotation:
            H_s = (u_2**2 - u_1**2) / (2 * self.g)
        else:
            H_s = u_2**2 / (2 * self.g)
        # Calculate ideal dynamic head
        H_d = u_2**2 / (2 * self.g)
        # Calculate ideal total and static heads
        H_total_ideal = H_s + H_d
        H_static_ideal = H_total_ideal + (v_inlet**2 - v_4**2) / (2 * self.g)

        # Calculate real static head. If the same prerotation assumption is used for analysis as in the case
        # of impeller design, use already calculated static head coefficient
        if no_prerotation == self.no_prerotation_design:
            static_head_coefficient = self.static_head_coefficient_BEP_Barske
        # If no prerotation was assumed for the design, but analysis assumes prerotation, then static head coefficient
        # needs to be increased
        elif self.no_prerotation_design and no_prerotation is False:
            static_head_coefficient = self.static_head_coefficient_BEP_Barske + (u_1 / u_2)**2
        # If prerotation was not assumed for the design, but analysis assume prerotation, then static head coefficient
        # needs to be decreased
        elif self.no_prerotation_design is False and no_prerotation:
            static_head_coefficient = self.static_head_coefficient_BEP_Barske - (u_1 / u_2)**2

        # Calculate real static head
        H_static_real = static_head_coefficient * u_2**2 / (2 * self.g)
        # Calculate real total head
        H_total_real = H_static_real + (v_4**2 - v_inlet**2) / (2 * self.g)
        # Calculate head loss
        H_loss = H_total_ideal - H_total_real

        # Calculate maximum velocity. Barske assumes maximum flow is reached when all pump head is equal to the dynamic
        # head in the throat
        v_max = np.sqrt(2 * self.g * (H_total_real + v_inlet**2 / (2 * self.g)))
        # If velocity in the throat is above the maximum value, the H-Q curve breaks down
        if v_3 > v_max:
            # H_loss becomes the total ideal head
            H_loss = H_total_ideal
            # Total real head becomes zero
            H_total_real = 0
            # Static head is then just the difference in dynamic heads
            H_static_real = (v_inlet**2 - v_4**2) / (2 * self.g)

        # Recalculate flow coefficients
        head_coefficient_static = 2 * self.g * H_static_real / u_2**2
        head_coefficient_total = 2 * self.g * H_total_real / u_2**2

        return (H_static_real, H_total_real, H_loss, H_total_ideal, H_static_ideal, head_coefficient_static,
                head_coefficient_total, H_s)

    def __calcualate_radial_clearance(self, D_2):
        """A method to calculate radial clearance of the impeller. From the empirical data from Barske (pg. 7 of
        "The Design of Open Impeller Centrifugal Pumps"), a linear equation for radial clearance was derived to do so.

         :param float or int D_2: Outer impeller diameter, m.

         :return: Radial clearance of the impeller, m.
         :rtype: float
         """
        return (0.008 * (D_2 / self.inch_to_m) + 0.072) * self.inch_to_m

    def analyse(self, fluid, mdot, RPM, p_upstream, T_upstream, analysis_method=None, K_factor=None, eta_losses=None,
                no_prerotation=None):
        """A method to analyze pump's performance for given operating conditions.

        :param Fluid fluid: Object representing pumped fluid.
        :param float or int mdot: Massflow through the pump, kg/s
        :param float or int RPM: Design Rotations Per Minute at BEP, 1 / minute
        :param float or int p_upstream: Upstream total (tank/reservoir) pressure, Pa
        :param float or int T_upstream: Upstream temperature, K
        :param string analysis_method: Method used for analysis. It should be either "Barske" or "Lock". If "Barske",
         a constant head coefficient is used to determine pump head. If "Lock", procedure from "A Forced Vortex Pump
         for High Speed, High Pressure, Low Flow Applications" by Lock will be used. If not given, a default analysis
          method which was used for sizing will be also used for analysis
        :param float or int eta_losses: Fraction of dynamic head lost in the diffuser, -. By default, the same value
         as used for the design when generating geometry with 'size_dimensions'.
        :param float or int K_factor: Factor for prerotation at zero flow as a fraction of inlet tip speed, -. Used for
            "Lock" option for analysis_method. By default, the same value
            as used for the design when generating geometry with 'size_dimensions'.
         :param bool no_prerotation: Boolean that determines whether no prerotation should be assumed for impeller head
            calculations when analysis_method is "Barske", True or False. By default, the same value as used for the
            design when generating geometry with 'size_dimensions'.

        :return: A dictionary with the analysis results.
        :rtype: dict
        """

        # If analysis_method is None, choose default one
        if analysis_method is None: analysis_method = self.default_analysis_method
        # If eta_losses is None, use the same value as the one used for design of the impeller
        if eta_losses is None: eta_losses = self.eta_losses_design
        # If K_factor is None, use the same value as the one used for design of the impeller
        if K_factor is None: K_factor = self.K_factor_design
        # If no_prerotation is None, use the same boolean as for the design
        if no_prerotation is None: no_prerotation = self.no_prerotation_design

        # Get fluid properties
        rho = fluid.get_density(p_upstream, T_upstream)  # kg/s
        kinematic_viscosity = fluid.get_kinematic_viscosity(p_upstream, T_upstream) # m^2 / s
        # Calculate volumetric flow at the inlet
        Q = mdot / rho  # m3 / s
        # First calculate flow speed and static pressure at the inlet.
        v_inlet = 4 * Q / (np.pi * self.D_inlet**2)
        p_inlet = p_upstream - 0.5 * rho * v_inlet**2
        # Calculate flow velocity and pressure at the impeller eye
        v_0 = 4 * Q / (np.pi * self.D_0**2)
        p_0 = p_upstream - 0.5 * rho * v_0 ** 2
        # Then calculate u_1, v_1m and v_1ax
        omega = RPM * 2 * np.pi / 60 # rad / s
        u_1 = omega * self.D_1 / 2 # m/s
        v_1ax = 4 * Q / (np.pi * self.D_1**2) # m/s
        v_1m = Q / self.A_1 # m/s
        w_1 = np.sqrt(u_1**2 + v_1m**2) # m/s
        # Calculate u_2
        u_2 = omega * self.D_2 / 2 # m/s
        # Calculate w_2
        w_2 = Q / self.A_2 # m/s
        # Calculate v_3 and v_4
        v_3 = 4 * Q / (np.pi * self.D_3 ** 2) # m/s
        v_4 = 4 * Q / (np.pi * self.D_4 ** 2) # m/s

        # Calculate paddle power to overcome friction for the impeller
        P_f_impeller = self.__calculate_friction_power(rho, kinematic_viscosity, RPM) # W
        
        # Calculate flow coefficient
        flow_coefficient_inlet = v_1ax / u_1 # -
        flow_coefficient_outlet = v_3 / u_2 # -

        # Calculate impeller ideal heads and coefficients depending on the method chosen
        if analysis_method == "Lock":
            (H_static_real, H_total_real, H_loss, H_total_ideal, H_static_ideal, head_coefficient_static,
             head_coefficient_total, _) = self.__analysis_Lock(u_2, u_1, Q, fluid, p_inlet, T_upstream, eta_losses, K_factor)
            # H_s is obtained from Barske, since it is more certain approach
            H_s = self.__analysis_Barske(u_1, u_2, v_inlet, v_3, v_4, no_prerotation)[-1]
        elif analysis_method == "Barske":
            (H_static_real, H_total_real, H_loss, H_total_ideal, H_static_ideal, head_coefficient_static,
             head_coefficient_total, H_s) = self.__analysis_Barske(u_1, u_2, v_inlet, v_3, v_4, no_prerotation)
        else:
            raise ValueError("analysis_method must be 'Lock' or 'Barske'")

        # Calculate p_2. To get difference between static head at the inlet and static head at the outlet, dynamic head
        # at the inlet must be added to static head due to forced vortex in the impeller
        p_2 = p_inlet + functions.get_dP_from_H(fluid, H_s + v_inlet**2 / (2 * self.g), p_inlet, T_upstream) # Pa
        # Calculate p_4.
        dp = functions.get_dP_from_H(fluid, H_static_real, p_inlet, T_upstream) # Pa
        p_4 = p_inlet + dp # Pa
        p_total_outlet = p_4 + 0.5 * rho * v_4**2 # Pa
        dp_total = p_total_outlet - p_upstream # Pa

        # Determine forces. First determine momentum force.
        F_momentum = mdot * v_0 # N

        # Now determine hydraulic axial force. Procedure and nomenclature is based on section 9.2 from 4th edition of
        # "Centrifugal Pumps" by Gulich, tables 9.1, 9.2 and 9.3. First k_0 will be calculated. k is ratio of tangential
        # fluid velocity to circumferential blade velocity. Impeller radius is obtained for convenience.
        r_2 = self.D_2 / 2 # m
        # There is a step in the casing if hub diameter is equal to the impeller diameter. Thickness and radius of that
        # step are t_ax and r_w respectively.
        if self.D_hub == self.D_2:
            # If hub does not overlap with that step, t_ax will equal t_hub to maintain constant axial clearance
            if self.s_ax > self.t_0: t_ax = self.t_0 # m
            # If hub overlaps with that step, t_ax is taken as equal to axial clearance
            else: t_ax = self.s_ax # m
            # r_w is impeller radius plus hub radial clearance
            r_w = r_2 + self.s_rad_hub # m
            # k_0 can be now calculated
            k_0 = 1 / (1 + (r_w / r_2)**2 * np.sqrt((r_w / r_2) + 5 * (t_ax / r_2))) # -
        # There are open sidewall gaps if hub diameter is smaller than impeller diameter. In this case, t_ax is zero
        # and r_w is equal to the radius of the hub.
        elif self.D_hub < self.D_2:
            # Hub ends before diameter, hence r_w is used instead of r_2 in the formula for k_0. It should evaluate to
            # 0.5 in such case.
            k_0 = 0.5 # -

        k_av = k_0 # -

        # Beyond hub forces are balanced, so hub diameter is used instead of impeller diameter in axial force
        # calculations. First obtain static pressure increase at the hub end assuming it varies linearly. It should vary
        # quadratically, but linear variation is what is assumed in Gulich in T9.3.1. In any case, linear variation
        # gives conservative results.
        p_hub = (p_2 - p_0) * (self.D_hub / self.D_2) + p_0 # Pa
        delta_p_hub = p_hub - p_0 # Pa
        # Also get blade velocity there
        u_hub = omega * self.D_hub / 2 # m/s

        # Hydraulic axial force can be now calculated
        dummy_2a = np.pi / 4 * self.D_hub**2
        dummy_2b = delta_p_hub * (1 - (self.D_shaft / self.D_hub)**2)
        dummy_2c = rho / 4 * (k_av * u_hub * (1 - (self.D_shaft / self.D_hub)**2))**2
        dummy_2d = delta_p_hub / 2 * (1 - (self.D_1 / self.D_hub)**2)**2
        F_hydraulic = dummy_2a * (dummy_2b - dummy_2c - dummy_2d) # N
        # Now calculate total axial force
        F_ax = F_hydraulic - F_momentum # N

        # Now calculate radial forces. Procedure is based on Table 9.7 from 4th edition of "Centrifugal Pumps" by
        # Gulich. First get flow at BEP at given RPM and calculte q_star
        Q_opt = self.A_3 * self.flow_coefficient_BEP * u_2 # m3/s
        q_star = Q / Q_opt # -
        # k_R0 is conservatively chosen to be 0.1, while a is 0.18. Both are values given by Gulich.
        k_R0 = 0.1 # -
        a = 0.18 # -
        # Now calculate k_R and radial static force.
        k_R = k_R0 * (1 + q_star + a * q_star**2) # -
        # Calculate total impeller width at the outlet
        if self.D_hub == self.D_2: L_2_total = self.L_2 + self.t_0 # m
        elif self.D_hub < self.D_2: L_2_total = self.L_2 # m
        # Calculate static radial force
        F_rad_static = k_R * rho * self.g * H_total_real * self.D_2 * L_2_total # N
        # Calcualte dynamic radial force. k_R_dyn is conservatively assumed to be 0.05 which is the value given by
        # Gulich
        k_R_dyn = 0.05 # -
        F_rad_dynamic = k_R_dyn * rho * self.g * H_total_real * self.D_2 * L_2_total # N
        # Calculate total peak radial force
        F_rad = F_rad_static + F_rad_dynamic # N

        # Calculate pressure at the shaft. This time assume quadratic variation and forced vortex
        # (more realistic results).
        p_hub = (p_2 - p_0) * (self.D_hub / self.D_2)**2 + p_0 # Pa
        r_hub = self.D_hub / 2 # m
        r_shaft = self.D_shaft / 2 # m
        p_shaft = p_hub - rho / 2 * (u_hub * k_av)**2 * (1 - (r_shaft / r_hub)**2) # Pa

        # Calculate total friction power
        P_f_total = P_f_impeller # W
        # Calculate useful pump power
        P_h_useful = mdot * H_total_real * self.g # W
        # Calculate pump power lost to hydraulic losses
        P_h_losses = mdot * H_loss * self.g # W
        # Calculate total hydraulic pump power
        P_h_total = mdot * H_total_ideal * self.g # W
        # Calculate total pump power
        P_total = P_h_total + P_f_total # -
        # Calculate pump efficiency
        eta_static = mdot * H_static_real * self.g / P_total # -
        eta_total = P_h_useful / P_total # -
        # Calculate outlet temperature. The calculated values are capped at those corresponding to 1e-3 kg/s, due to
        # unrealistically high values otherwise. Values can be still to high, but this avoids the failure.
        mdot_dummy = max(mdot, 1e-3)
        T_outlet = T_upstream + (P_h_losses + P_f_total) / (mdot_dummy * fluid.get_specific_heat(p_inlet, T_upstream)) # K

        # Determine torque acting on the impeller
        Torque = P_total / omega # N*m

        # Pack results into dictionary and return it
        analysis_results = {"method": analysis_method, "fluid": fluid, "RPM": RPM, "omega": omega, "mdot": mdot,
                            "dp": dp, "dp_total": dp_total, "Q": Q, "H_total_real": H_total_real,
                            "H_static_real": H_static_real, "H_losses": H_loss, "H_total_ideal": H_total_ideal,
                            "H_static_ideal": H_static_ideal, "flow_coefficient_inlet": flow_coefficient_inlet,
                            "flow_coefficient_outlet": flow_coefficient_outlet,
                            "static_head_coefficient": head_coefficient_static, "P_h_useful": P_h_useful,
                            "P_h_losses": P_h_losses, "P_h_total": P_h_total, "P_total": P_total,
                            "P_f_impeller": P_f_impeller, "P_f_total": P_f_total,
                            "eta_total": eta_total, "eta_static": eta_static, "eta_losses": eta_losses,
                            "T_upstream": T_upstream, "p_upstream": p_upstream, "rho": rho, "p_inlet": p_inlet,
                            "p_0": p_0, "p_shaft": p_shaft, "p_hub": p_hub, "p_2": p_2, "p_4": p_4,
                            "p_total_outlet": p_total_outlet, "v_inlet": v_inlet, "v_0": v_0, "u_1": u_1, "v_1ax": v_1ax,
                            "v_1m": v_1m, "w_1": w_1, "u_2": u_2, "w_2": w_2, "v_3": v_3, "v_4": v_4, "T_4": T_outlet,
                            "F_ax": F_ax, "F_rad": F_rad, "Torque": Torque}
        return analysis_results

    def verify_design(self):
        """A method to verify the design of the Barske pump. It will print whether all conditions are satisfied and set
         the condition flags in object properties. Impeller guidelines are from 'The Design of Open Impeller Centrifugal
          Pumps' by Barske."""
        print("\nVeryfying pump geometry. Recommended guidelines were taken from literature and semi-empirical data,\n"
              "so it may happen that not all can be satisfied. These guidelines should be used for informative purpose.\n")
        # Verify if axial velocity at the inlet is within recommended range
        self.design_checks["eye_velocity"] = \
            bool(5 * self.feet_to_m <= self.analysis_results_design["v_0"] <= 12 * self.feet_to_m)
        if not self.design_checks["eye_velocity"]:
            print(f"Impeller eye diameter v0 is {self.analysis_results_design['v_0']:.3f} m/s."
                  f" It should be between {5*self.feet_to_m:.3f} m/s and {12*self.feet_to_m:.3f} m/s.")
        # Verify if inner blade velocity is smaller than recommended by Barske
        self.design_checks["LE_velocity"] = bool(self.analysis_results_design["u_1"] <= 150 * self.feet_to_m)
        if not self.design_checks["LE_velocity"]:
            print(f"Inner blade speeed u1 is {self.analysis_results_design['u_1']:.3f} m/s."
                  f" It should be <= {150 * self.feet_to_m:.3f} m/s.")
        # Verify if diameter ratio is greater than recommended by Barske
        self.design_checks["diameter_ratio"] = bool(self.D_2 / self.D_1 > 1.5)
        if not self.design_checks["diameter_ratio"]:
            print(f"Diameter ratio D2/D1 is {self.D_2 / self.D_1:.3f}."
                  f"It should be > 1.5.")
        # Verify axial widths. Barske did not take into account blade thicknesses, which are taken into account here.
        # The first was derived from L_1 => 0.25 D_1 recommendation from Barske,
        # while the second was derived from L_2 => L_1 * D_1 / D_2
        L_1_recommended = (0.25 * np.pi * self.D_1**2) / (np.pi * self.D_1 - self.n_blades * self.t_1)
        self.design_checks["LE_width"] = bool(self.L_1 >= L_1_recommended)
        if not self.design_checks["LE_width"]:
            print(f"Axial width L1 at impeller inlet is {self.L_1 * 1000:.3f} mm."
                  f" It should be above or equal {L_1_recommended * 1000:.3f} mm.")
        # Calculate the minimum outlet width using the actual blade and splitter thicknesses
        outlet_blockage = self.n_blades * self.t_2
        if self.splitter_blades:
            outlet_blockage += self.n_blades * self.t_splitter_2
        L_2_recommended = self.A_1 / (self.D_2 * np.pi - outlet_blockage)
        self.design_checks["TE_width"] = bool(self.L_2 >= L_2_recommended)
        if not self.design_checks["TE_width"]:
            print(f"Axial width L2 at impeller outlet is {self.L_2 * 1000:.3f} mm."
                  f" It should be above or equal {L_2_recommended * 1000:.3f} mm.")
        # Verify that clearances are within recommended values. From Barske (pg. 7 of "The Design of Open Impeller
        # Centrifugal Pumps"), axial clearance should be 1% of D_2, but should not be greater than
        # 0.04 inch of larger pumps. From the empirical data from the same page, a linear equation for radial
        # clearance is derived, however it is just a guideline.
        s_ax_max = min(0.01 * self.D_2, 0.04 * self.inch_to_m) # m
        self.design_checks["axial_clearance"] = bool(self.s_ax <= s_ax_max)
        if not self.design_checks["axial_clearance"]:
            print(f"Axial clearance s_ax is {self.s_ax * 1000:.3f} mm."
                  f" It should be below {s_ax_max * 1000:.3f} mm.")
        s_rad_max = self.__calcualate_radial_clearance(self.D_2) # m
        self.design_checks["radial_clearance"] = bool(self.s_rad <= s_rad_max)
        if not self.design_checks["radial_clearance"]:
            print(f"Radial clearance s_rad is {self.s_rad * 1000:.3f} mm."
                  f" Semi-empirical data indicates it should be below {s_rad_max * 1000:.3f} mm.")
        self.design_checks["TE_width_clearance_ratio"] = bool(self.L_2 >= 3 * self.s_ax)
        if not self.design_checks["TE_width_clearance_ratio"]:
            print(f"Axial width L2 at impeller outlet is {self.L_2 * 1000:.3f} mm."
                  f" It should be greater than 3s_ax = {3 * self.s_ax * 1000:.3f} mm.")
        # According to Gulich, throat diameter should be a bit smaller than TE width
        self.design_checks["D_3"] = bool(self.D_3 <= self.L_2)
        if not self.design_checks["D_3"]:
            print(f"Throat diameter is {self.D_3 * 1000:.3f} mm."
                  f" It should be smaller than L_2 = {self.L_2 * 1000:.3f} mm.")

    def print_dimensions(self):
        """A method to print pump's dimensions in a GitHub-style table"""
        # Create a function for safe handling of None
        def safe(val, factor=1.0):
            return val * factor if val is not None else None

        # Create rows
        rows = [
            # Diameters
            ["D_inlet", self.D_inlet * 1e3, "mm", "Inlet pipe diameter"],
            ["D_0", self.D_0 * 1e3, "mm", "Impeller eye diameter"],
            ["D_1", self.D_1 * 1e3, "mm", "Impeller inlet diameter"],
            ["D_2", self.D_2 * 1e3, "mm", "Impeller outlet diameter"],
            ["D_3", self.D_3 * 1e3, "mm", "Diffuser throat diameter"],
            ["D_4", self.D_4 * 1e3, "mm", "Diffuser outlet diameter"],
            ["D_hub", self.D_hub * 1e3, "mm", "Hub diameter"],
            ["D_shaft", self.D_shaft * 1e3, "mm", "Shaft diameter"],
            ["D_1/D_0", self.D_1_over_D_0, "-", "Ratio of D1 to D0"],
            ["D_hub/D_1", self.D_hub_over_D_1, "-", "Ratio of D_hub to D1"],

            # Widths & lengths
            ["L_1", self.L_1 * 1e3, "mm", "Impeller inlet width"],
            ["L_2", self.L_2 * 1e3, "mm", "Impeller outlet width"],
            ["L_1/D_1", self.L_1_over_D_1, "-", "Ratio of L1 to D1"],
            ["L_diffuser", self.L_diffuser * 1e3, "mm", "Diffuser length"],

            # Areas
            ["A_1", self.A_1 * 1e6, "mm^2", "Impeller inlet flow area"],
            ["A_2", self.A_2 * 1e6, "mm^2", "Impeller outlet flow area"],
            ["A_3", self.A_3 * 1e6, "mm^2", "Diffuser throat area"],
            ["A_4", self.A_4 * 1e6, "mm^2", "Diffuser exit area"],
            ["A_4/A_3", self.A_4_over_A_3, "-", "Diffuser area ratio"],

            # Clearances
            ["s_ax", self.s_ax * 1e3, "mm", "Axial clearance"],
            ["s_rad", self.s_rad * 1e3, "mm", "Radial clearance"],
            ["s_rad_hub", safe(self.s_rad_hub, 1e3), "mm", "Radial clearance of the hub"],

            # Thicknesses
            ["t_0", self.t_0 * 1e3, "mm", "Impeller hub thickness"],
            ["t_1", self.t_1 * 1e3, "mm", "Blade thickness at inlet"],
            ["t_2", self.t_2 * 1e3, "mm", "Blade thickness at outlet"],

            # Angles
            ["alpha_0", self.alpha_0, "deg", "Blade forward edge angle"],
            ["alpha_1", self.alpha_1, "deg", "Blade backward edge angle"],
            ["alpha_2", self.alpha_2, "deg", "Radial blade sharpening angle"],
            ["alpha_diffuser", self.alpha_diffuser, "deg", "Diffuser full angle"],

            # Other
            ["n_blades", self.n_blades, "-", "Number of blades"],
            ["specific_speed", self.specific_speed, "m, RPM, m^3/s", "Specific speed (EU)"]
        ]

        # Add splitter dimensions when splitter blades are used
        if self.splitter_blades:
            rows.extend([
                ["D_splitter", self.D_splitter * 1e3, "mm", "Splitter blade inlet diameter"],
                ["D_splitter/D_1", self.D_splitter_over_D_1, "-", "Ratio of D_splitter to D_1"],
                ["t_splitter_1", self.t_splitter_1 * 1e3, "mm", "Splitter blade thickness at splitters' inlet"],
                ["t_splitter_2", self.t_splitter_2 * 1e3, "mm", "Splitter blade thickness at outlet"]
            ])

        # Print table
        print("\n GEOMETRY OF THE PUMP:")
        print(tabulate(
            rows,
            headers=["Parameter", "Value", "Unit", "Description"],
            tablefmt="github",
            floatfmt=".2f"
        ))

    def print_analysis_results(self, analysis_results):
        """A method to print results of the analysis

        :param dict analysis_results: Dictionary created with 'analyse' method.
        """

        # Create abberviation for the analysis results
        r = analysis_results

        # Create rows
        rows = [
            ["method", r["method"], "-", "Analysis method"],
            ["fluid", r["fluid"].name, "-", "Fluid used for analysis"],
            ["RPM", r["RPM"], "1/min", "Rotations per minute"],
            ["mdot", r["mdot"], "kg/s", "Mass flow rate"],
            ["Q", r["Q"]*1e3, "L/s", "Volumetric flow rate"],
            ["dp", r["dp"] * 1e-5, "bar", "Pressure rise across pump"],

            ["flow_coefficient_inlet", r["flow_coefficient_inlet"], "-", "Flow coefficient at impeller inlet"],
            ["flow_coefficient_outlet", r["flow_coefficient_outlet"], "-", "Flow coefficient at diffuser throat"],
            ["static_head_coefficient", r["static_head_coefficient"], "-", "Static head coefficient"],

            ["H_total_real", r["H_total_real"], "m", "Real total head"],
            ["H_static_real", r["H_static_real"], "m", "Real static head"],
            ["H_losses", r["H_losses"], "m", "Head losses"],
            ["H_total_ideal", r["H_total_ideal"], "m", "Ideal total head"],
            ["H_static_ideal", r["H_static_ideal"], "m", "Ideal static head"],

            ["P_h_useful", r["P_h_useful"] * 1e-3, "kW", "Useful hydraulic power"],
            ["P_h_losses", r["P_h_losses"] * 1e-3, "kW", "Hydraulic losses power"],
            ["P_h_total", r["P_h_total"] * 1e-3, "kW", "Total hydraulic power"],
            ["P_total", r["P_total"] * 1e-3, "kW", "Total pump power"],
            ["P_f_impeller", r["P_f_impeller"] * 1e-3, "kW", "Impeller friction power"],
            ["P_f_total", r["P_f_total"] * 1e-3, "kW", "Total friction power"],

            ["eta_total", r["eta_total"], "-", "Total efficiency"],
            ["eta_static", r["eta_static"], "-", "Static efficiency"],
            ["eta_losses", r["eta_losses"], "-", "Fraction of diffuser dynamic head lost"],

            ["T_upstream", r["T_upstream"], "K", "Upstream temperature"],
            ["rho", r["rho"], "kg/m^3", "Fluid density"],

            ["p_upstream", r["p_upstream"] * 1e-5, "bar", "Upstream pressure"],
            ["p_inlet", r["p_inlet"] * 1e-5, "bar", "Pressure at impeller eye"],
            ["p_0", r["p_0"] * 1e-5, "bar", "Pressure at impeller eye"],
            ["p_shaft", r["p_shaft"] * 1e-5, "bar", "Pressure at impeller shaft"],
            ["p_hub", r["p_hub"] * 1e-5, "bar", "Pressure at the edge of impeller hub"],
            ["p_2", r["p_2"] * 1e-5, "bar", "Pressure at impeller outlet"],
            ["p_4", r["p_4"] * 1e-5, "bar", "Pressure at diffuser outlet"],

            ["v_inlet", r["v_inlet"], "m/s", "Velocity in inlet pipe"],
            ["v_0", r["v_0"], "m/s", "Velocity at impeller eye"],

            ["u_1", r["u_1"], "m/s", "Leading edge blade velocity"],
            ["v_1ax", r["v_1ax"], "m/s", "Absolute axial velocity at impeller inlet"],
            ["v_1m", r["v_1m"], "m/s", "Absolute meridional velocity at impeller inlet"],
            ["w_1", r["w_1"], "m/s", "Relative velocity at impeller inlet"],

            ["u_2", r["u_2"], "m/s", "Blade tip velocity"],
            ["w_2", r["w_2"], "m/s", "Relative flow velocity at the impeller outlet"],
            ["v_3", r["v_3"], "m/s", "Velocity in diffuser throat"],
            ["v_4", r["v_4"], "m/s", "Velocity at diffuser outlet"],

            ["T_4", r["T_4"], "K", "Outlet temperature"],

            ["F_ax", r["F_ax"], "N", "Axial force"],
            ["F_rad", r["F_rad"], "N", "Radial force"],
            ["Torque", r["Torque"], "N*m", "Pump torque"],
        ]

        # Postprocess table so that numbers to three digital places are printed
        def format(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else x
        rows = [[format(v) for v in row] for row in rows]

        # Print table
        print("\n ANALYSIS RESULTS:")
        print(tabulate(
            rows,
            headers=["Parameter", "Value", "Unit", "Description"],
            tablefmt="github",
            floatfmt=".3f"
        ))

    def __get_blade_plot_profiles(self):
        """Calculate the top and side blade profiles for geometry plots."""
        # Convert the blade dimensions to millimeters and angles to radians
        r1 = self.D_1 * 1e3 / 2 # mm
        r2 = self.D_2 * 1e3 / 2 # mm
        r_hub = self.D_hub * 1e3 / 2 # mm
        L1 = self.L_1 * 1e3 # mm
        L2 = self.L_2 * 1e3 # mm
        t0 = self.t_0 * 1e3 # mm
        t1 = self.t_1 * 1e3 # mm
        t2 = self.t_2 * 1e3 # mm
        alpha1 = np.deg2rad(self.alpha_1) # rad
        alpha2 = np.deg2rad(self.alpha_2) # rad

        # Create base blade geometry at reference angular position, where x-axis is aligned with radial coordinates.
        # First get coordinates of the leading edge of the blade.
        blade_inner_point_1 = np.array([r1, -t1 / 2])
        # Coordinates of one of the LE points must be found from the sharpening angle. To do so, calculate slope of
        # blade suction surface
        blade_surface_slope = ((t2 / 2) - (t1 / 2)) / (r2 - r1)
        # Calculate slope of sharpened leading edge
        sharpening_slope = np.tan((np.pi/2) - alpha2)
        # Solve for intersection
        blade_inner_point_2_x = r1 + (t1 / (sharpening_slope - blade_surface_slope))
        blade_inner_point_2_y = (-t1 / 2) + sharpening_slope * (blade_inner_point_2_x - r1)
        blade_inner_point_2 = np.array([
            blade_inner_point_2_x,
            blade_inner_point_2_y
        ])
        # Get coordinates of the outer points
        blade_outer_point_1 = np.array([r2, t2 / 2])
        blade_outer_point_2 = np.array([r2, -t2 / 2])
        # Assemble blade polygon
        blade_polygon = np.array([
            blade_inner_point_1,
            blade_inner_point_2,
            blade_outer_point_1,
            blade_outer_point_2,
            blade_inner_point_1
        ])
        # Splitter leading edges are straight and tangent to D_splitter, without sharpening
        splitter_polygon = None
        if self.splitter_blades:
            r_splitter = self.D_splitter * 1e3 / 2 # mm
            t_splitter_1 = self.t_splitter_1 * 1e3 # mm
            t_splitter_2 = self.t_splitter_2 * 1e3 # mm
            splitter_polygon = np.array([
                [r_splitter, -t_splitter_1 / 2],
                [r_splitter, t_splitter_1 / 2],
                [r2, t_splitter_2 / 2],
                [r2, -t_splitter_2 / 2],
                [r_splitter, -t_splitter_1 / 2]
            ])

        # Create blade geometry. First create blade polygon for the case when hub diameter does not cover the whole
        # impeller
        if self.D_hub != self.D_2:
            # Compute leading edge points.
            blade_side_point_0 = np.array([r1, t0 + L1])
            blade_side_point_1 = np.array([r1, t0])
            # Compute intermediate points at the hub
            blade_side_point_2 = np.array([r_hub, t0])
            blade_side_point_3 = np.array([r_hub, 0])
            # Now trailing edge points
            y2 = np.tan((np.pi / 2) - alpha1) * (r2 - r1)
            blade_side_point_4 = np.array([r2, y2])
            blade_side_point_5 = np.array([r2, y2 + L2])
            # Create blade polygon
            blade_side_polygon = np.array([
                blade_side_point_0,
                blade_side_point_1,
                blade_side_point_2,
                blade_side_point_3,
                blade_side_point_4,
                blade_side_point_5,
                blade_side_point_0
            ])
        # Now the option when hub covers the whole impeller
        elif self.D_hub == self.D_2:
            # Compute leading edge points.
            blade_side_point_0 = np.array([r1, t0 + L1])
            blade_side_point_1 = np.array([r1, t0])
            # Now trailing edge points
            blade_side_point_2 = np.array([r2, t0])
            blade_side_point_3 = np.array([r2, t0 + L2])
            # Create blade polygon
            blade_side_polygon = np.array([
                blade_side_point_0,
                blade_side_point_1,
                blade_side_point_2,
                blade_side_point_3,
                blade_side_point_0,
            ])

        # Trim the main blade side profile at D_splitter to give the splitter a straight leading edge
        splitter_side_polygon = None
        if self.splitter_blades:
            splitter_side_points = []
            for point_1, point_2 in zip(blade_side_polygon[:-1], blade_side_polygon[1:]):
                if point_1[0] >= r_splitter:
                    splitter_side_points.append(point_1)
                # Interpolate where a profile edge crosses the splitter inlet radius
                if (point_1[0] < r_splitter) != (point_2[0] < r_splitter):
                    fraction = (r_splitter - point_1[0]) / (point_2[0] - point_1[0])
                    splitter_side_points.append(point_1 + fraction * (point_2 - point_1))
            # Start the splitter profile at its upper leading edge, as for a main blade
            splitter_side_points.insert(0, splitter_side_points.pop())
            splitter_side_points.append(splitter_side_points[0])
            splitter_side_polygon = np.array(splitter_side_points)

        # Return the closed main and optional splitter profiles
        return blade_polygon, blade_side_polygon, splitter_polygon, splitter_side_polygon

    def __get_casing_plot_profile(self, blade_side_polygon):
        """Calculate the meridional casing profile in millimeters for geometry plots."""
        # Get the casing radii and clearances in millimeters
        r1 = self.D_1 * 1e3 / 2 # mm
        r_eye = self.D_0 * 1e3 / 2 # mm
        r_hub = self.D_hub * 1e3 / 2 # mm
        r_shaft = self.D_shaft * 1e3 / 2 # mm
        s_ax = self.s_ax * 1e3 # mm
        s_rad = self.s_rad * 1e3 # mm
        l_shaft = 2 * s_ax # mm

        # Get the blade edges next to the casing from either side profile
        leading_edge = blade_side_polygon[0]
        lower_trailing_edge, upper_trailing_edge = blade_side_polygon[-3:-1]

        # Now create pump casing geometry. First consider option if there is no step change in geometry.
        if self.D_hub != self.D_2:
            # Points next to the shaft
            casing_point_0 = np.array([r_shaft * 1.05, -l_shaft])
            casing_point_1 = np.array([r_shaft * 1.05, -s_ax])
            # Point at the inner diameter of the blades
            casing_point_2 = np.array([r1, -s_ax])
            # Points next to the trailing edge
            casing_point_3 = np.array([lower_trailing_edge[0], lower_trailing_edge[1] - s_ax])
            casing_point_4 = np.array([lower_trailing_edge[0] + s_rad, lower_trailing_edge[1] - s_ax])
            casing_point_5 = np.array([upper_trailing_edge[0] + s_rad, upper_trailing_edge[1] + s_ax])
            casing_point_6 = np.array([upper_trailing_edge[0], upper_trailing_edge[1] + s_ax])
            # Point next to the leading edge
            casing_point_7 = np.array([leading_edge[0], leading_edge[1] + s_ax])
            # Point next to the impeller eye
            y_eye = casing_point_6[1] + (casing_point_7[1] - casing_point_6[1]) * (r_eye - casing_point_6[0]) \
                    / (casing_point_7[0] - casing_point_6[0])
            casing_point_8 = np.array([r_eye, y_eye])
            # Create pump casing polygon
            casing_polygon = np.array([
                casing_point_0,
                casing_point_1,
                casing_point_2,
                casing_point_3,
                casing_point_4,
                casing_point_5,
                casing_point_6,
                casing_point_7,
                casing_point_8
            ])
        # Now consider pump geometry if there is a step change in the casing
        elif self.D_hub == self.D_2:
            # Points at the shaft
            casing_point_0 = np.array([r_shaft * 1.05, -l_shaft])
            casing_point_1 = np.array([r_shaft * 1.05, -s_ax])
            # Now get radius of the intermediate points
            r_w = r_hub + self.s_rad_hub * 1e3
            # Get intermediate points
            casing_point_2 = np.array([r_w, -s_ax])
            casing_point_3 = np.array([r_w, lower_trailing_edge[1] - s_ax])
            # Get casing points around trailing edge
            casing_point_4 = np.array([lower_trailing_edge[0], lower_trailing_edge[1] - s_ax])
            casing_point_5 = np.array([lower_trailing_edge[0] + s_rad, lower_trailing_edge[1] - s_ax])
            casing_point_6 = np.array([upper_trailing_edge[0] + s_rad, upper_trailing_edge[1] + s_ax])
            casing_point_7 = np.array([upper_trailing_edge[0], upper_trailing_edge[1] + s_ax])
            # Point next to the leading edge
            casing_point_8 = np.array([leading_edge[0] + s_rad, leading_edge[1] + s_ax])
            # Point next to the impeller eye
            y_eye = casing_point_7[1] + (casing_point_8[1] - casing_point_7[1]) * (r_eye - casing_point_7[0]) \
                    / (casing_point_8[0] - casing_point_7[0])
            casing_point_9 = np.array([r_eye, y_eye])
            # Create pump casing polygon
            casing_polygon = np.array([
                casing_point_0,
                casing_point_1,
                casing_point_2,
                casing_point_3,
                casing_point_4,
                casing_point_5,
                casing_point_6,
                casing_point_7,
                casing_point_8,
                casing_point_9
            ])

        # Return the open casing profile between the shaft and the impeller eye
        return casing_polygon

    @staticmethod
    def __get_diffuser_plot_position(r2, r_volute, D3):
        """Calculate the diffuser throat center in the impeller plane."""
        # Position the throat tangent to the impeller and put its inner point on the volute radius
        x0 = -r2
        y0 = np.sqrt(r_volute**2 - (x0 + D3 / 2)**2)
        return x0, y0

    @staticmethod
    def __rotate_blade_points(points, angle):
        """Rotate blade coordinates in the impeller plane by the given angle."""
        # Calculate the rotation matrix and apply it to each point
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return points @ rotation_matrix.T

    def plot_geometry_2D(self):
        """Plot top and side views of the pump geometry in millimeters."""
        # Get the top and side blade profiles
        blade_polygon, blade_side_polygon, splitter_polygon, splitter_side_polygon = self.__get_blade_plot_profiles()

        # Convert pump dimensions from meters to millimeters
        D0 = self.D_0 * 1e3 # mm
        D2 = self.D_2 * 1e3 # mm
        D3 = self.D_3 * 1e3 # mm
        D4 = self.D_4 * 1e3 # mm
        D_hub = self.D_hub * 1e3 # mm
        D_shaft = self.D_shaft * 1e3 # mm
        L_diffuser = self.L_diffuser * 1e3 # mm
        t0 = self.t_0 * 1e3 # mm
        s_ax = self.s_ax * 1e3 # mm
        s_rad = self.s_rad * 1e3 # mm
        # Get radiuses
        r2 = D2 / 2 # mm
        r_eye = D0 / 2 # mm
        r_hub = D_hub / 2 # mm
        r_shaft = D_shaft / 2
        r_volute = (D2 + 2 * s_rad) / 2 # mm
        # First create figure with two subplots - one subplot will be for top view and another for side view
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Set the blade pitch and angular offset
        blade_pitch_angle = 2 * np.pi / self.n_blades
        rotation_offset = blade_pitch_angle / 2

        # Plot top view geometry
        ax_top = axes[0]
        theta = np.linspace(0, 2 * np.pi, 300)
        ax_top.plot(r_eye * np.cos(theta), r_eye * np.sin(theta), label="Impeller eye", linestyle='--')
        ax_top.plot(r_hub * np.cos(theta), r_hub * np.sin(theta), label="Impeller hub")
        ax_top.plot(r_volute * np.cos(theta), r_volute * np.sin(theta), label="Pump volute")
        # Plot all blades
        for i in range(self.n_blades):
            angle = rotation_offset + i * blade_pitch_angle
            rotated_blade = self.__rotate_blade_points(blade_polygon, angle)
            if i == 0:
                ax_top.plot(rotated_blade[:, 0], rotated_blade[:, 1], label="Impeller blades")
            else:
                ax_top.plot(rotated_blade[:, 0], rotated_blade[:, 1])

        # Plot one splitter blade halfway between each pair of main blades
        if self.splitter_blades:
            for i in range(self.n_blades):
                angle = rotation_offset + (i + 0.5) * blade_pitch_angle
                rotated_splitter = self.__rotate_blade_points(splitter_polygon, angle)
                if i == 0:
                    ax_top.plot(rotated_splitter[:, 0], rotated_splitter[:, 1], label="Splitter blades")
                else:
                    ax_top.plot(rotated_splitter[:, 0], rotated_splitter[:, 1])

        # Create diffuser geometry
        x0, y0 = self.__get_diffuser_plot_position(r2, r_volute, D3)
        throat_point_1 = np.array([x0 - (D3 / 2), y0])
        throat_point_2 = np.array([x0 + (D3 / 2), y0])
        outlet_point_1 = np.array([x0 - (D4 / 2), y0 + L_diffuser])
        outlet_point_2 = np.array([x0 + (D4 / 2), y0 + L_diffuser])
        # Create diffuser polygon
        diffuser_polygon = np.array([
            throat_point_2,
            outlet_point_2,
            outlet_point_1,
            throat_point_1
        ])
        # Plot the diffuser
        ax_top.plot(diffuser_polygon[:, 0], diffuser_polygon[:, 1], label="Pump diffuser")

        # Set graph options
        ax_top.set_title("Top view")
        ax_top.set_xlabel("x (mm)")
        ax_top.set_ylabel("y (mm)")
        ax_top.grid(True, which="major")
        ax_top.grid(True, which="minor", linestyle=":")
        ax_top.minorticks_on()
        ax_top.legend(loc="upper right")
        ax_top.set_aspect("equal")

        # Plot side view geometry now in the next subfigure
        ax_side = axes[1]

        # Plot the blade side profile
        ax_side.plot(blade_side_polygon[:, 0], blade_side_polygon[:, 1], label="Impeller blade")

        # Plot the splitter side profile if present
        if self.splitter_blades:
            ax_side.plot(splitter_side_polygon[:, 0], splitter_side_polygon[:, 1],
                         label="Splitter blade", linestyle="--")

        # Create impeller hub geometry
        hub_point_0 = np.array([0, 0])
        hub_point_1 = np.array([r_hub, 0])
        hub_point_2 = np.array([r_hub, t0])
        hub_point_3 = np.array([0, t0])
        # Create hub polygon
        hub_polygon = np.array([
            hub_point_0,
            hub_point_1,
            hub_point_2,
            hub_point_3
        ])
        # Plot it
        ax_side.plot(hub_polygon[:, 0], hub_polygon[:, 1], label="Impeller hub")

        # Create shaft geometry
        shaft_point_0 = np.array([0, 0])
        shaft_point_1 = np.array([r_shaft, 0])
        # Determine length of shaft to plot
        l_shaft = 2 * s_ax
        shaft_point_2 = np.array([r_shaft, - l_shaft])
        # Create shaft polygon
        shaft_polygon = np.array([
            shaft_point_0,
            shaft_point_1,
            shaft_point_2
        ])
        # Plot it
        ax_side.plot(shaft_polygon[:, 0], shaft_polygon[:, 1], label="Impeller shaft")

        # Plot the pump casing
        casing_polygon = self.__get_casing_plot_profile(blade_side_polygon)
        ax_side.plot(casing_polygon[:, 0], casing_polygon[:, 1], label="Pump casing")

        # Set graph options
        ax_side.set_title("Side view")
        ax_side.set_xlabel("r (mm)")
        ax_side.set_ylabel("z (mm)")
        ax_side.grid(True, which="major")
        ax_side.grid(True, which="minor", linestyle=":")
        ax_side.minorticks_on()
        ax_side.set_aspect("equal")
        ax_side.legend(loc="upper right")

        # Show figure
        plt.tight_layout()
        plt.show()

    def plot_geometry_3D(self):
        """Plot the impeller, volute and diffuser in millimeters in an interactive Plotly browser window.

        Call after sizing the pump with 'size_dimensions' or assigning geometry with 'assign_dimensions'.
        """
        # Get the blade profiles and the meridional casing profile used in the 2D plot
        blade_polygon, blade_side_polygon, splitter_polygon, splitter_side_polygon = self.__get_blade_plot_profiles()
        casing_polygon = self.__get_casing_plot_profile(blade_side_polygon)

        # Create a Plotly figure and add triangulated surfaces with component labels on hover
        fig = go.Figure()

        # Add each component mesh to the Plotly figure, skipping collapsed triangles.
        def add_mesh(vertices, faces, name, color, opacity=1.0, legendgroup=None, showlegend=True):
            # Filter triangles whose vertices coincide when a ring or blade edge collapses
            triangles = vertices[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            area_tolerance = np.finfo(float).eps * np.max(np.ptp(vertices, axis=0))**2
            faces = faces[np.linalg.norm(normals, axis=1) > area_tolerance]
            # Keep dimensions in millimeters and group the hub and blades under one legend entry
            fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name, color=color,
                                    opacity=opacity, legendgroup=legendgroup, showlegend=showlegend,
                                    lighting=dict(ambient=0.65, diffuse=0.75, specular=0.2),
                                    hovertemplate=name + "<br>x: %{x:.2f} mm<br>y: %{y:.2f} mm"
                                                       "<br>z: %{z:.2f} mm<extra></extra>"))

        # Triangulate between successive sections for blades and surfaces of revolution.
        def get_ring_faces(sections):
            # Wrap the last vertex of each section back to the first to close the seam
            section_size = sections.shape[1]
            start = sections[:-1].ravel()
            next_corner = np.roll(sections[:-1], -1, axis=1).ravel()
            next_section = start + section_size
            return np.vstack((np.column_stack((start, next_corner, next_section)),
                              np.column_stack((next_corner, next_corner + section_size, next_section))))

        # Build a closed mesh from the top and side profiles of a main or splitter blade.
        def get_blade_mesh(plan_polygon, side_polygon):
            # The two plan edges give the tangential bounds at each radial station
            pressure_edge = plan_polygon[[0, 3]]
            suction_edge = plan_polygon[:3]
            r_inlet, r_outlet = pressure_edge[:, 0]
            corner_radii = suction_edge[:, 0]
            # The side profile gives the upper edge and the lower segments on either side of the hub
            upper_edge = side_polygon[[0, -2]]
            lower_edge = side_polygon[1:-2]
            rings = []
            for point_1, point_2 in zip(lower_edge[:-1], lower_edge[1:]):
                # Keep the vertical shoulder at the hub, including its two different axial positions
                if point_1[0] == point_2[0] and r_inlet <= point_1[0] <= r_outlet:
                    radii = np.array([point_1[0], point_2[0]])
                    z_lower = np.array([point_1[1], point_2[1]])
                else:
                    # Limit the segment to the plan profile and split it at the sharpening corner
                    r_start = max(r_inlet, point_1[0])
                    r_end = min(r_outlet, point_2[0])
                    if r_start >= r_end:
                        continue
                    interior_corners = corner_radii[(r_start < corner_radii) & (corner_radii < r_end)]
                    radii = np.concatenate(([r_start], interior_corners, [r_end]))
                    z_lower = np.interp(radii, [point_1[0], point_2[0]], [point_1[1], point_2[1]])
                # Build a four-corner section at each radius from the tangential and axial bounds
                y_pressure = np.interp(radii, pressure_edge[:, 0], pressure_edge[:, 1])
                y_suction = np.interp(radii, suction_edge[:, 0], suction_edge[:, 1])
                z_upper = np.interp(radii, upper_edge[:, 0], upper_edge[:, 1])
                x = np.broadcast_to(radii[:, None], (len(radii), 4))
                y = np.column_stack((y_pressure, y_suction, y_suction, y_pressure))
                z = np.column_stack((z_lower, z_lower, z_upper, z_upper))
                rings.append(np.stack((x, y, z), axis=-1))

            # Triangulate the four sides between sections and cap the leading and trailing edges
            vertices = np.concatenate(rings).reshape(-1, 3)
            sections = np.arange(len(vertices)).reshape(-1, 4)
            inlet_faces = sections[0][[[0, 2, 1], [0, 3, 2]]]
            outlet_faces = sections[-1][[[0, 1, 2], [0, 2, 3]]]
            faces = np.vstack((get_ring_faces(sections), inlet_faces, outlet_faces))
            return vertices, faces

        # Sample each revolution at the same azimuths and join the final point to the first
        azimuths = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        cosine, sine = np.cos(azimuths), np.sin(azimuths)

        # Join circular sections and plot the resulting hub, volute or diffuser mesh.
        def add_ring_mesh(x, y, z, name, color, opacity=1.0, legendgroup=None):
            # Assemble the coordinates at each azimuth and join the circular sections
            vertices = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
            sections = np.arange(vertices.shape[0] * len(azimuths)).reshape(-1, len(azimuths))
            add_mesh(vertices.reshape(-1, 3), get_ring_faces(sections), name, color, opacity, legendgroup)

        # Revolve the hub profile from the axis out to its rim and back to close both faces
        r_hub = self.D_hub * 1e3 / 2 # mm
        t0 = self.t_0 * 1e3 # mm
        hub_radii = np.array([0, r_hub, r_hub, 0])[:, None]
        hub_z = np.array([0, 0, t0, t0])[:, None]
        add_ring_mesh(hub_radii * cosine, hub_radii * sine, hub_z, "Impeller", "blue", legendgroup="Impeller")

        # Rotate the main blade mesh at the same pitch and offset as in the 2D plot
        blade_pitch = 2 * np.pi / self.n_blades
        blade_rows = [(blade_polygon, blade_side_polygon, "Impeller blade", 0.5)]
        # Put each splitter halfway between two main blades if splitters are present
        if self.splitter_blades:
            blade_rows.append((splitter_polygon, splitter_side_polygon, "Splitter blade", 1.0))
        for plan, side, name, offset in blade_rows:
            vertices, faces = get_blade_mesh(plan, side)
            for i in range(self.n_blades):
                angle = (i + offset) * blade_pitch
                blade_xy = self.__rotate_blade_points(vertices[:, :2], angle)
                add_mesh(np.column_stack((blade_xy, vertices[:, 2])), faces, f"{name} {i + 1}", "blue",
                         legendgroup="Impeller", showlegend=False)

        # Revolve the open casing profile around the rotation axis to form the transparent volute
        casing_radii, casing_z = casing_polygon.T[:, :, None]
        add_ring_mesh(casing_radii * cosine, casing_radii * sine, casing_z, "Pump volute", "grey", opacity=0.2)

        # Position the throat as in the top view and center the diffuser axially on the impeller outlet
        r2 = self.D_2 * 1e3 / 2 # mm
        r_volute = r2 + self.s_rad * 1e3 # mm
        D3 = self.D_3 * 1e3 # mm
        D4 = self.D_4 * 1e3 # mm
        x0, y0 = self.__get_diffuser_plot_position(r2, r_volute, D3)
        z0 = np.mean(blade_side_polygon[-3:-1, 1])
        # Draw only the cone wall from the throat to the outlet as a separate surface
        diffuser_radii = np.array([D3 / 2, D4 / 2])[:, None]
        diffuser_y = np.array([y0, y0 + self.L_diffuser * 1e3])[:, None]
        add_ring_mesh(x0 + diffuser_radii * cosine, diffuser_y, z0 - diffuser_radii * sine,
                      "Pump diffuser", "grey", opacity=0.2)

        # Keep the same scale on all axes and allow orbiting and scroll zoom in the browser
        fig.update_layout(title="Pump 3D geometry", template="plotly_white",
                          scene=dict(aspectmode="data", dragmode="orbit", xaxis_title="x (mm)",
                                     yaxis_title="y (mm)", zaxis_title="z (mm)",
                                     camera=dict(eye=dict(x=1.55, y=1.9, z=1.55))),
                          margin=dict(l=15, r=15, b=25, t=45))
        fig.show(renderer="browser", config=dict(scrollZoom=True, displaylogo=False, responsive=True))
