* Case study III: Phenotype clustering of ICU patients with sepsis based on 
* temperature trajectories
********************************************************************************

clear all
set more off
set linesize 120

* Set working directory
cd CHANGE_THIS_PATH

* Create output directory for results
capture mkdir "stata"
capture mkdir "plots"

* Start logging
log using "stata/gbtm.log", replace text

* Define datasets to process
local datasets "mimic_iv mimic_iii amsterdamumcdb"

foreach dataset in `datasets' {

display "Starting processing dataset: `dataset'"

********************************************************************************
* 1. DATA PREPARATION
********************************************************************************
* Import pre-pivoted CSV data
import delimited "data/TEMPERATURE_GBTM_`dataset'.csv", clear varnames(1)

* Data is already in wide format with:
* - id: patient identifier
* - temp_0 to temp_72: standardized temperature at each hour
* - time_0 to time_72: time values for each measurement

* Rename id to patient_id for consistency
rename id patient_id

* Display data structure
describe
display _newline "Number of patients: " _N

* Rename variables to match traj expectations (remove underscore before number)
* traj expects temp0, temp1, ... not temp_0, temp_1, ...
forvalues i = 0/72 {
    capture rename temp_`i' temp`i'
    capture rename time_`i' time`i'
}

* Count non-missing temperature observations per patient
egen n_obs = rownonmiss(temp*)
summarize n_obs, detail
display _newline "Non-missing temperature observations per patient:"
display "  Mean: " r(mean)
display "  Median: " r(p50)
display "  Min: " r(min)
display "  Max: " r(max)
drop n_obs

* Display summary statistics for temperature variables
display _newline(2) "Summary Statistics for Standardized Temperature (first 72 hours):"
summarize temp*, separator(0)

* Final check of data structure
describe
display _newline "Data is ready for trajectory modeling"

********************************************************************************
* 2. GROUP-BASED TRAJECTORY MODELING
********************************************************************************
display _newline(2) "{hline 80}"
display "STEP 2: Group-Based Trajectory Modeling"

* Note: The traj command uses the pre-pivoted wide format data where:
* - Each row represents one patient (N = " _N " patients)
* - var() specifies dependent variables: temp0, temp1, ... temp72
*   (standardized temperature at hours 0 through 72)
* - indep() specifies independent variables: time0, time1, ... time72
*   (corresponding time values for each measurement)
* - order() specifies polynomial order for each group's trajectory
*   (0=intercept, 1=linear, 2=quadratic, 3=cubic)
* - model(cnorm) for censored normal distribution with min() and max()
*   (uses actual data range calculated above)
* - The command automatically creates _traj_Group and _traj_ProbG* variables
*   after each model estimation

********************************************************************************
* 2.1 FOUR-GROUP QUADRATIC MODEL
********************************************************************************
* Model: 4 groups, quadratic (order 2)
display _newline "Model: 4 groups, all quadratic"
traj, model(cnorm) var(temp*) indep(time*) ///
    order(2 2 2 2) min(-6) max(72) detail
scalar bic_4g_quad_`dataset' = e(BIC_n_subjects)
display "  BIC (4-group quadratic): " bic_4g_quad_`dataset'

********************************************************************************
* 3. DETAILED RESULTS
********************************************************************************
display _newline(2) "{hline 80}"
display "STEP 3: Detailed Results"

* Plot trajectories with confidence intervals from the model
trajplot, xtitle("Hours Since ICU Admission") ///
    ytitle("Temperature (standardized)")
graph export "stata/`dataset'_traj_4group_quadratic.png", replace width(1200)

* Save group assignment probabilities
* The traj command automatically generates _traj_Group and _traj_ProbG* variables
* Check if _traj_Group exists in the dataset
capture confirm variable _traj_Group
if _rc != 0 {
    display as error "Warning: _traj_Group variable not found. This may occur if the model didn't converge."
    display as error "Group assignments will not be saved."
}
else {
    preserve
    keep patient_id _traj_Group _traj_ProbG*
    duplicates drop
    export delimited using "stata/`dataset'_trajectory_group_assignments.csv", replace
    display _newline "Group assignments saved to: stata/`dataset'_trajectory_group_assignments.csv"
    restore
}

* Display group membership statistics
display _newline(2) "Group Membership:"

tab _traj_Group
display _newline "Proportion in each group:"
tab _traj_Group, missing

* Display average posterior probabilities
display _newline(2) "Average Posterior Probability of Group Membership:"
display "{hline 80}"
display "Each patient should have high probability for their assigned group"
display "(Average posterior probability > 0.7 indicates good classification)"
display "{hline 80}"

* Calculate and display average posterior probability for each group
quietly levelsof _traj_Group, local(groups)
foreach g of local groups {
    quietly summarize _traj_ProbG`g' if _traj_Group == `g'
    display "Group `g': Mean posterior probability = " %5.3f r(mean) ///
            " (n=" r(N) ")"
}

display "Finished processing dataset: `dataset'"

}

********************************************************************************
* 4. FINAL OUTPUT
********************************************************************************
display _newline(2) "{hline 80}"
display "ANALYSIS COMPLETE"
display _newline "Output files:"
foreach dataset in `datasets' {
    display "  Dataset `dataset':"
    display "    - stata/`dataset'_traj_4group_quadratic.png - Trajectory plot"
    display "    - stata/`dataset'_trajectory_group_assignments.csv - Group assignments"
}

log close
