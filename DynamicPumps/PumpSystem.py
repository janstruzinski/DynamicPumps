import numpy as np

class PumpSystem:
    def __init__(self, stages):
        """A class to analyse pump system consisting of one or multiple stages like inducers, partial emission or full
         emission impellers.

         :param list stages: A list containing objects representing pump stages.
         """

        # Stage list
        self.stages = stages
        # RPM - mdot sweep results array
        self.sweep_results = None
        # Constants
        self.g = 9.80665  # gravitational acceleration m/s^2

    def analyse(self, fluid, mdot, RPM, p_upstream, T_upstream):
        """A method to analyse connected pump stages.

        :param Fluid fluid: Object representing pumped fluid.
        :param float or int mdot: Massflow through the pump, kg/s
        :param float or int RPM: Design Rotations Per Minute at BEP, 1 / minute
        :param float or int p_upstream: Upstream total (tank/reservoir) pressure, Pa
        :param float or int T_upstream: Upstream temperature, K

        :return: A tuple with the analysis results. First dictionary holds analysis results for the whole pump system.
            Second dictionary holds list with analysis results for individual stages.
        :rtype: tuple
        """

        # Create upstream conditions
        T_0 = T_upstream
        p_0 = p_upstream

        # Create analysis_results list to store analysis_results for each stage
        stages_results = []

        # Create variables for pump system static and total heads, static and total pressure rises, useful and total
        # powers
        H_static = 0 # m
        H_total = 0 # m
        dp_static = 0 # Pa
        dp_total = 0 # Pa
        P_h_useful = 0 # W
        P_total = 0 # W

        # Iterate over each stage and get results
        for stage in self.stages:
            analysis_results = stage.analyse(fluid, mdot, RPM, p_0, T_0)
            stages_results.append(analysis_results)
            # Update upstream conditions for the next stage
            p_0 = analysis_results["p_total_outlet"]
            T_0 = analysis_results["T_4"]
            # Update pump system variables
            H_static += analysis_results["H_static_real"]
            H_total += analysis_results["H_total_real"]
            dp_static += analysis_results["dp"]
            dp_total += analysis_results["dp_total"]
            P_h_useful += analysis_results["P_h_useful"]
            P_total += analysis_results["P_total"]

        # Update outlet variables
        p_total_outlet = p_0
        T_outlet = p_0

        # Calculate total to total and static to total efficiency for the whole pump system
        eta_total = P_h_useful / P_total # -
        eta_static = mdot * H_static * self.g / P_total # -

        # Assemble pump system results dictionary
        analysis_results = {"fluid": fluid.name, "RPM": RPM, "mdot": mdot, "H_static_real": H_static,
                            "H_total_real": H_total, "dp": dp_static, "dp_total": dp_total, "P_h_useful": P_h_useful,
                            "P_total": P_total, "eta_total": eta_total, "eta_static": eta_static,
                            "T_upstream": T_upstream, "p_upstream": p_upstream, "T_outlet": T_outlet,
                            "p_total_outlet": p_total_outlet}

        # Return results
        return analysis_results, stages_results

    def sweep_over_RPM_and_mdot(self, fluid, p_upstream, T_upstream, mdot_range, RPM_range):
        """A method to sweep over RPM and mdot to get results at different operating conditions for the pump system.

        :param Fluid fluid: Object representing pumped fluid.
        :param float or int p_upstream: Upstream total (tank/reservoir) pressure, Pa
        :param float or int T_upstream: Upstream temperature, K
        :param np.ndarray or list mdot_range: A list of mass flows (kg/s) over which to do sweep
        :param np.ndarray or list RPM_range: A list of RPM (-) over which to do sweep
        """

        # Create an array of operating conditions. Each row represents different RPM, while each column represents
        # different massflow.

        # Create an array to store results for these operating conditions.

        # Sweep over mdot and RPM ranges and get results.

    def extract_sweep_results(self, flattened=False):
        """A method to extract pressure rise, efficiency and shaft power arrays from sweep results over operating
         conditions.

        :param bool flattened: Boolean whether arrays should be returned flattened. If False, 2D arrays will be returned
            where rows correspond to different RPM and columns correspond to different mdot. If True, these arrays will
            be flattened. By default, False.
        """

        # Extract operating conditions (RPM & mdot) array

        # Get total and static pressure increase arrays

        # Get total and static pump head arrays

        # Get total and static efficiency arrays

        # Get power array

        # Return the arrays

    def plot_pump_map(self):
        ...