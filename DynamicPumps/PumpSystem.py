import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import ContourSet

class PumpSystem:
    def __init__(self, stages):
        """A class to analyse pump system consisting of one or multiple stages like inducers, partial emission or full
         emission impellers.

         :param list stages: A list containing objects representing pump stages.
         """

        # Stage list
        self.stages = stages
        # RPM - mdot sweep results list. First object is meshgrid array with RPM, second one is meshgrid array with
        # mdot, third one is meshgrid array with results. Each element of the last array is a tuple - its first element
        # is a dictionary with the results for the whole system, while its second element is a dictionary with the
        # results for the individual stages.
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

        # Calculate volumetric flow
        Q = mdot / fluid.get_density(p_0, T_0)

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
        analysis_results = {"fluid": fluid.name, "RPM": RPM, "mdot": mdot, "Q": Q, "H_static_real": H_static,
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

        # Create meshgrid of operating conditions. Each row represents different RPM, while each column represents
        # different massflow.
        mdot_range = np.array(mdot_range)
        RPM_range = np.array(RPM_range)
        RPM_grid, mdot_grid = np.meshgrid(RPM_range, mdot_range, indexing='ij')

        # Create an array to store results for these operating conditions.
        results_grid = np.empty_like(RPM_grid, dtype=object)

        # Sweep over mdot and RPM ranges and get results.
        for i in range(len(RPM_range)):
            for j in range(len(mdot_range)):
                RPM = RPM_grid[i, j]
                mdot = mdot_grid[i, j]
                tuple_results = self.analyse(fluid, mdot, RPM, p_upstream, T_upstream)
                results_grid[i, j] = tuple_results

        # Assign sweep results to class property
        self.sweep_results = [RPM_grid, mdot_grid, results_grid]

    def extract_sweep_results(self, flattened=False):
        """A method to extract pressure rise, efficiency and shaft power arrays from sweep results over operating
         conditions.

        :param bool flattened: Boolean whether arrays should be returned flattened. If False, 2D arrays will be returned
            where rows correspond to different RPM and columns correspond to different mdot. If True, these arrays will
            be flattened. By default, False.
        """

        # Extract operating conditions (RPM & mdot) and results array
        RPM_grid = self.sweep_results[0]
        mdot_grid = self.sweep_results[1]
        results_grid = self.sweep_results[2]
        shape = RPM_grid.shape

        # Initialize arrays to store data
        dp_total = np.empty(shape)
        dp_static = np.empty(shape)
        H_total = np.empty(shape)
        H_static = np.empty(shape)
        eta_total = np.empty(shape)
        eta_static = np.empty(shape)
        P_total = np.empty(shape)
        Q = np.empty(shape)

        # Fill arrays
        for i in range(shape[0]):
            for j in range(shape[1]):
                analysis_results = results_grid[i, j][0]
                # Get total and static pressure increase arrays
                dp_total[i, j] = analysis_results["dp_total"]
                dp_static[i, j] = analysis_results["dp"]
                # Get total and static pump head arrays
                H_total[i, j] = analysis_results["H_total_real"]
                H_static[i, j] = analysis_results["H_static_real"]
                # Get total and static efficiency arrays
                eta_total[i, j] = analysis_results["eta_total"]
                eta_static[i, j] = analysis_results["eta_static"]
                # Get shaft power array
                P_total[i, j] = analysis_results["P_total"]
                # Get volumetric flow array
                Q[i, j] = analysis_results["Q"]

        # Flatten if requested
        if flattened:
            return (RPM_grid.ravel(), mdot_grid.ravel(), dp_total.ravel(), dp_static.ravel(), H_total.ravel(),
                    H_static.ravel(), eta_total.ravel(), eta_static.ravel(), P_total.ravel(), Q.ravel())

        # Otherwise return as is
        return RPM_grid, mdot_grid, dp_total, dp_static, H_total, H_static, eta_total, eta_static, P_total, Q

    def plot_pump_map(self, QH_map=False, x_min=None, y_min=None, y_max=None, no_contours=12,
                      max_efficiency_point=False, additional_isolines=False):
        """A method to plot pump performance map (either mdot-dP or Q-H) with total efficiency contours.

        :param bool QH_map: If True, plots volumetric flow rate Q (L/s) versus static head H (m).
            If False, plots mass flow rate mdot (kg/s) versus static pressure rise dp (bar).
        :param int or float x_min: Optional lower limit for x-axis.
        :param int or float y_min: Optional lower limit for y-axis.
        :param int or float y_max: Optional higher limit for y-axis.
        :param int no_contours: Number of total efficiency contours. By default, 12.
        :param bool max_efficiency_point: Boolean whether maximum efficiency point and its value should be shown for
            each RPM curve. By default, True.
        :param bool additional_isolines: Boolean whether additional efficiency isolines should be plotted from the
            maximum efficiency points towards higher RPM curves. By default, False.
        """

        # Extract sweep data
        RPM_grid, mdot_grid, dp_total, dp_static, H_total, H_static, eta_total, eta_static, P_total, Q = \
            self.extract_sweep_results(flattened=False)

        # Select axes
        if QH_map:
            x = Q * 1000  # L/s
            y = H_static
            x_label = "Volumetric flow rate [L/s]"
            y_label = "Static head [m]"
        else:
            x = mdot_grid
            y = dp_static / 1e5  # bar
            x_label = "Mass flow rate [kg/s]"
            y_label = "Static pressure rise [bar]"

        # Create figure
        fig, ax = plt.subplots()

        # Plot RPM curves (truncate at first y <= 0)
        n_RPM = RPM_grid.shape[0]
        max_efficiency_indices = []
        max_efficiency_points = []
        for i in range(n_RPM):
            RPM_value = RPM_grid[i, 0]
            x_i = x[i, :]
            y_i = y[i, :]
            # Find first index where y <= 0
            invalid_idx = np.where(y_i <= 0)[0]
            if invalid_idx.size > 0:
                cut_idx = invalid_idx[0]
                x_i = x_i[:cut_idx + 1]
                y_i = y_i[:cut_idx + 1]
            # Plot RPM curves
            RPM_line, = ax.plot(x_i, y_i, label=f"RPM = {RPM_value:.0f}")
            # Get maximum efficiency point within the operating region
            max_efficiency_idx = None
            if max_efficiency_point or additional_isolines:
                eta_i = np.where((y[i, :] > 0) & np.isfinite(eta_total[i, :]), eta_total[i, :], np.nan)
                if not np.all(np.isnan(eta_i)):
                    max_efficiency_idx = np.nanargmax(eta_i)
            max_efficiency_indices.append(max_efficiency_idx)
            if max_efficiency_point and max_efficiency_idx is not None:
                max_efficiency_points.append([x[i, max_efficiency_idx], y[i, max_efficiency_idx],
                                              eta_i[max_efficiency_idx], RPM_line.get_color()])

        # Efficiency contours. First mask non-operating region (only where y > 0)
        valid_mask = y > 0
        eta_masked = np.where(valid_mask, eta_total, np.nan)
        # Create contour plot
        levels = np.linspace(0, np.nanmax(eta_masked), no_contours)
        contour = ax.contour(x, y, eta_masked, colors='0.5',  linewidths=0.6, levels=levels)
        # Label contours
        ax.clabel(contour, inline=True, fontsize=8, fmt="η = %.3f")

        # Plot additional isolines starting from maximum efficiency points towards higher RPM curves
        if additional_isolines:
            RPM_order = np.argsort(RPM_grid[:, 0])
            for i in range(n_RPM - 1):
                start_RPM_idx = RPM_order[i]
                start_efficiency_idx = max_efficiency_indices[start_RPM_idx]
                if start_efficiency_idx is None:
                    continue
                efficiency = eta_total[start_RPM_idx, start_efficiency_idx]
                isoline_x = [x[start_RPM_idx, start_efficiency_idx]]
                isoline_y = [y[start_RPM_idx, start_efficiency_idx]]
                # Find the same efficiency on each higher RPM curve
                for next_RPM_idx in RPM_order[i + 1:]:
                    next_max_efficiency_idx = max_efficiency_indices[next_RPM_idx]
                    if next_max_efficiency_idx is None:
                        continue
                    # Search from the maximum efficiency point towards lower flow rates
                    for j in range(next_max_efficiency_idx, 0, -1):
                        eta_1 = eta_total[next_RPM_idx, j]
                        eta_2 = eta_total[next_RPM_idx, j - 1]
                        if not np.isfinite(eta_1) or not np.isfinite(eta_2):
                            continue
                        if min(eta_1, eta_2) <= efficiency <= max(eta_1, eta_2):
                            if eta_1 == eta_2:
                                fraction = 0
                            else:
                                fraction = (efficiency - eta_1) / (eta_2 - eta_1)
                            isoline_x.append(x[next_RPM_idx, j] +
                                             fraction * (x[next_RPM_idx, j - 1] - x[next_RPM_idx, j]))
                            isoline_y.append(y[next_RPM_idx, j] +
                                             fraction * (y[next_RPM_idx, j - 1] - y[next_RPM_idx, j]))
                            break
                # Connect interpolated points and label the isoline
                if len(isoline_x) > 1:
                    isoline = ContourSet(ax, [efficiency], [[np.column_stack((isoline_x, isoline_y))]],
                                         colors='0.5', linewidths=0.6)
                    ax.clabel(isoline, inline=True, fontsize=8, fmt="η = %.3f")

        # Plot and label maximum efficiency point for each RPM curve
        for point in max_efficiency_points:
            ax.plot(point[0], point[1], marker='o', color=point[3])
            ax.annotate("η = %.3f" % point[2], (point[0], point[1]), xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', color='0.5', bbox=dict(facecolor='none', edgecolor='none'))

        # Axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Pump Map")
        # Grid
        ax.grid(True, which='major')
        ax.grid(True, which='minor', linestyle='--', alpha=0.5)
        ax.minorticks_on()
        # Axis limits
        if x_min is not None:
            ax.set_xlim(left=x_min)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        elif y_min is None:
            ax.set_ylim(bottom=0)
        if y_max is not None:
            ax.set_ylim(top=y_max)
        # Legend
        ax.legend()
        # SHow plot
        plt.tight_layout()
        plt.show()
