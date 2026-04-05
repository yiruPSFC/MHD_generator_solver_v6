# Characterization of a $\mathbf{5 0 k W}$ Inductively Coupled Plasma Torch for Testing of Ablative Thermal Protection Materials 

Benton R. Greene ${ }^{1}$, Noel T. Clemens ${ }^{2}$, and Philip L. Varghese ${ }^{3}$<br>University of Texas at Austin, Austin, TX, 78705<br>Stanley A. Bouslog ${ }^{4}$ and Steven V. Del Papa ${ }^{5}$<br>NASA Johnson Space Center, Houston, TX, 77058


#### Abstract

With the development of new manned spaceflight capabilities including NASA's Orion capsule and the Space-X Dragon capsule, there is a renewed importance of understanding the dynamics of ablative thermal protection systems. To this end, a new inductively coupled plasma torch facility is being developed at UT-Austin. The torch operates on argon and/or air at plasma powers up to 25 kW for input power up to 50 kW . In the present configuration the flow exits from a low-speed subsonic nozzle and the hot plume is characterized using slug calorimetry and emission spectroscopy. Measurements using emission spectroscopy have indicated that the torch is capable of producing an air plasma with a temperature between $5,000 \mathrm{~K}$ and $8,000 \mathrm{~K}$ depending on the power and flow settings and an argon plasma with a temperature of approximately $12,000 \mathrm{~K}$. The temperature falls off from the central peak value by approximately $1,000 \mathrm{~K}$ at a radius of 8 mm . The facility operation envelope was determined, and heat flux was measured for selected points within the envelope using both a slug calorimeter and a Gardon gauge heat flux sensor. The torch was found to induce a stagnation point heat flux of between 90 and $225 \mathrm{~W} / \mathrm{cm}^{2}$. A small asymmetry of unknown cause which increases with increasing mass flow rate was found in the radial variation of heat flux.


## Nomenclature

| I | = Intensity of a spectral line |
| :--- | :--- |
| $n$ | = number density |
| $A_{u l}$ | = Transition probability |
| $g$ | = degeneracy of electronic state |
| $\lambda$ | = wavelength |
| $\varepsilon$ | = energy of electronic state |
| $T_{B}$ | = Boltzmann temperature |
| $k_{B}$ | = Boltzmann constant |
| $V_{a}$ | = Anode voltage of plasma torch |
| $\dot{m}_{\text {air }}$ | = mass flow rate of air |
| $m$ | = mass of slug calorimeter |
| $c_{p}$ | $=$ specific heat |
| $\dot{T}$ | $=$ time rate of change of slug calorimeter temperature |
| $\dot{q}$ | = heat flux |

[^0]$A \quad=$ surface area of slug calorimeter
$Z=$ distance from nozzle exit in streamwise direction

## I. Introduction

Developing effective thermal protection systems for spacecraft entering a planetary atmosphere is a challenging task. The system must be as light as possible while still effectively dissipating the heat load of atmospheric entry and insulating the spacecraft against it. Additionally, flight testing such a system is prohibitively expensive if not impossible, and the designers must be highly certain the system will work before it is flight tested.

Predicting the performance of a given thermal protection system requires high-fidelity models of how the surface pyrolizes and ablates in the superhot environment behind a hypersonic bow shock. These models, in turn, require extensive knowledge of chemical kinetic rate coefficients in extreme temperature flows; however, such information is not available in many cases or not particularly accurate.

Various ground testing facilities can be used to partially reconstruct the reentry environment and provide empirical data against which computational models can be compared and updated. One stalwart of ablation research has been the arcjet, which heats a high pressure stream of air to temperatures of several thousand Kelvin using a high-power electrical discharge before expanding the superheated flow through a supersonic nozzle. These arc-heated hypersonic facilities were used in the 1960s [1] and are still widely used today [2]-[4]. However, the use of an electrical arc discharge to heat the air has the drawback that the flow is contaminated with metal ions from the exposed electrodes.

Inductively coupled plasma (ICP) devices, which use an oscillating magnetic field within an inductor to couple energy into a partially ionized gas, do not have the problem of flow contamination. First described in 1961 by Reed [5], these devices typically involve a high-power oscillator circuit driven by a high-voltage DC supply to create an AC signal in the range of $400 \mathrm{kHz}-10 \mathrm{MHz}$ which is fed through a low-turn inductor coil surrounding a quartzencased plasma chamber. Since a metal electrode is never in contact with the gas, no metal ions are deposited into the flow, resulting in a much cleaner flow for high-enthalpy pyrolysis research. However, ICP torches have the disadvantage that they do not work as well as arc-heated wind tunnels at high flow rates or high pressures.

In recent years, several institutions have utilized ICP facilities to study how materials react to high temperature flows. The most powerful of these is the 1.2 MW plasmatron at the Von Kármán Institute [6], which has been in operation since the late 90s. Other inductive facilities exist in Germany [7], France [8], and Vermont in the US [9], and range in power from 30 kW to 180 kW .

The current work involves characterizing the plasma produced by a new ICP torch facility at The University of Texas at Austin at various gas flow and input power settings. The data will be used to create an empirical model of how to set the torch to achieve desired test conditions. These measurements were performed using standard measurement techniques like emission spectroscopy to determine a temperature profile across the radius of the torch plume, a slug calorimeter to measure the cold-wall heat flux to a test article, and a Gardon gauge to measure the heat

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-02.jpg?height=660&width=844&top_left_y=1273&top_left_x=1013)
Figure 1. Photo of torch in operation (without insertion mechanism in place)

flux.

## II. Experiments

## A. Plasma Torch Facility

Experiments were performed using a 50 kW inductively coupled argon/air plasma torch facility at the JJ Pickle Research Campus at The University of Texas at Austin. The torch was developed and built for UT by Applied Plasma Technologies and uses a $6 \mathrm{MHz}, 12 \mathrm{kV}$ AC signal to deposit up to 25 kW of power into up to 70 slpm of air and/or argon. The nozzle of the torch is 30 mm in diameter, and the plasma chamber is 60 mm in diameter and 250 mm in
length. The chamber is enclosed in a double-wall quartz tube and is water cooled. The inductor coil around the tube is also water cooled to prevent arcing between its turns.

Plasma is stabilized inside the chamber using one of two stabilization modes: direct (or forward) vortex or reverse vortex gas injection. Forward vortex stabilization is a common mode of plasma stabilization in ICP devices with net gas flow through the plasma chamber. First described by Reed [5], it works by injecting a swirling gas on the upstream end of the plasma chamber. The resulting flow rotation drives a recirculation zone in the axis of the chamber, which aids plasma propagation. Additionally, the centripetal acceleration of the rotating fluid combined with the buoyancy of the hot plasma causes the plasma to float toward the axis of the chamber, helping to insulate the chamber walls and reduce parasitic heat losses to the wall cooling system.

Reverse vortex stabilization, first applied to ICPs by Gutsol [10], injects the swirling gas into the nozzle end of the plasma chamber. A step change in diameter between the inner diameter of the chamber and the nozzle aperture prevents the immediate exit of the injected gas from the torch. Instead, the injected gas flows around the outer radius of the chamber to the far end and rebounds up through the central axis of the chamber and out of the exit nozzle. In this way, the flow next to the chamber walls consists of cold gas unprocessed by the magnetic field, further insulating the walls from the hot core flow. Additionally the extra layer of swirling flow restricts the size of the core flow further, causing all gas to flow through the high temperature zone without inducing regions of recirculation. This method of plasma stabilization has been shown to be more efficient than forward vortex stabilization [10].

A unique feature of this ICP system is that the torch head is connected to the power supply using a flexible tether, which allows the torch head to be moved within a limited range. This configuration allows the torch plume to be translated and thus to aid with optical diagnostic measurements.

## 1. Traverse System

The torch head is mounted to a three-axis traverse to allow optical diagnostic equipment to inspect any arbitrary point in the plasma flow. The traverse uses precision ball screw actuators and servo motors to position the torch head to within 0.1 mm over a 150 mm travel range on any axis, and can be programmed to perform a timed move synchronized with a data collection system to automatically capture data from multiple points in the flow during a single run.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-03.jpg?height=917&width=1207&top_left_y=1401&top_left_x=466)
Figure 2. Model of plasma torch with insertion mechanism and 3-axis traverse.

## 2. Probe/Test Article Insertion Mechanism

Physical probes and test articles are held in the plasma plume using a custom-built water cooled insertion mechanism. The mechanism has two arms to allow both a heat flux probe and a material specimen to be mounted at the same time and rotated into the plume smoothly and swiftly. In this way, the heat flux can be accurately measured immediately before inserting a test article. The arms are driven by a servo motor and can be accurately positioned to the center of the plume or scanned across the diameter of the plume to get a radial profile of flow properties. The whole insertion mechanism structure is rigidly mounted to the torch head, so that scanning the plasma plume through a laser diagnostic interrogation volume does not affect the relative positioning of the test article or physical probe within the plasma. Figure 2 shows a model of how the torch, traversing table and test article insertion mechanism fit together in the assembly.

## B. Experimental Techniques

## 3. Heat Flux

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-04.jpg?height=483&width=1345&top_left_y=836&top_left_x=389)
Figure 3. Slug calorimeter image and assembly diagram.

Cold-wall stagnation point heat flux is measured using both a Gardon gauge heat flux sensor and a slug calorimeter. The slug calorimeter, pictured in Figure 3, consists of a slug of copper of known dimensions and heat capacity with a type K thermocouple embedded in the back surface. The slug is surrounded by a copper housing, from which it is insulated by a thin air gap such that the heat conduction within the slug can be considered one dimensional. The calorimeter is inserted into the plasma plume for 2 to 4 seconds and the slope of the resulting temperature curve is measured. The time rate of change of the temperature of the slug, $\dot{T}$, can be related to the stagnation point heat flux, $\dot{q}$, per the relation

$$
\begin{equation*}
\dot{q}=\frac{m c_{p}}{A} \dot{T} \tag{1}
\end{equation*}
$$

where $m, c_{p}$, and $A$ are the mass, specific heat, and exposed surface area of the slug, respectively.
The slug used had a mass of 15.797 g and a specific heat of $404 \mathrm{~J} / \mathrm{kg} \cdot \mathrm{K}$. The exposed surface of the slug had a spherical contour to ensure that the heat flux is approximately constant over the entire face and has a total surface area of $2 \mathrm{~cm}^{2}$.

Because copper is a catalyst for oxygen reactions, it is likely that the measured heat flux values will have contributions from the catalyzed reactions on the surface of the slug. We do not attempt to characterize this in the present work, but to ensure that all measurements are uniformly catalytic, the surface of the slug was polished after every two insertions with Wenol metal polish. This prevented buildup of any deposits on the surface that might affect the catalycity and therefore render the measurements inconsistent.

The Gardon gauge, pictured in Figure 4, consists of a copper body with a constantan foil disc at the stagnation point. The copper body is water cooled, providing the cold junction of the thermocouple junction between the copper and the constantan. The gauge outputs a mV level signal proportional to the stagnation point heat flux. The gauges used in the present work are manufactured by Medtherm and come with a NIST traceable calibration curve from the

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-05.jpg?height=394&width=1075&top_left_y=252&top_left_x=525)
Figure 4. (Left) Large diameter Gardon gauge (Right) small diameter Gardon Gauge

manufacturer which is used to convert the output signal to heat flux. Two sizes of Gardon gauge are used in the current study: a large gauge designed to be the same size and shape as the test articles to more closely match the heat flux that a test article will see and a small-diameter probe for measuring radial variation in stagnation point heat flux across the diameter of the plume. The large probe measures 30 mm in diameter and has a 30 mm radius spherical front surface. The small probe measures 5 mm in diameter.

The probes are attached to the insertion arms via adapters such that the column is a constant diameter for at least three probe diameters downstream of the tip of the probe in order to avoid disturbing the measurement.

The output signals from each of these probes are captured by a $5 \mathrm{kHz}, 24$ bit analog to digital converter card in a National Instruments PXIe 4310 data acquisition system. Careful measures were taken to ensure the probes were adequately grounded and shielded from the RF environment of the torch power supply. An additional 20 Hz software filter was applied to the slug calorimeter data to further reduce noise.

## 4. Emission Spectroscopy

The temperature of the plasma is measured using an emission spectroscopy system. This system uses an OceanOptics HR4000 spectrometer to measure the emission spectrum between 200 nm and 1100 nm , giving the sensor an effective spectral resolution of 0.44 $\mathrm{nm} / \mathrm{px}$. Light from the plasma is coupled into the spectrometer using a UV transmissive fiber optic cable. The captured spectrum is calibrated against an Ocean Optics LS-1-CAL tungsten calibration lamp to account for transmission losses in the optical system and efficiency of the CCD sensor. The spectral line broadening due to the optics of the spectrometer was estimated to be about 1.2 nm (FWHM), and was determined by recording the atomic emission line spectra from a low-pressure mercury vapor lamp.

To obtain the radial variation of the plasma temperature, the collection beam was limited to a 1 mm wide aperture with a pencil-like field of view using an adjustable iris and a 150 mm focal length lens placed several focal lengths away from the torch plume, as pictured in Figure 5. Spectra were captured at $y$ locations across the diameter of the plume at 1 mm intervals to obtain a spectral intensity vs position, $I(\lambda ; y)$. These intensity measurements were then transformed using an Abel inversion process to obtain the spectral intensity vs radial location, $I(\lambda ; r)$. The Abel inversion technique described in Villareal and Varghese [11] was used. The Abel inverted spectra were then used to

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-05.jpg?height=689&width=426&top_left_y=1271&top_left_x=1393)
Figure 5. Diagram of emission spectroscopy collection optics

calculate an electronic temperature as a function of plume radius.

## III. Results and Discussion

Several measurement campaigns have been carried out to probe the capabilities of the new facility and correlate various output parameters of interest to the user-visible controls. The torch has two primary user inputs: the DC voltage supplied to the anode of the oscillator circuit vacuum tube, $V_{a}$, and the mass flow rate of the test gas through the plasma chamber, $\dot{m}_{\text {air }}$. Most of the data in the present paper is presented with respect to one or both of these inputs.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-06.jpg?height=643&width=1026&top_left_y=271&top_left_x=506)
Figure 6. Operational envelope using air as a test gas.

Most of the campaigns were carried out using air as the test gas, as the torch circuitry has been optimized for coupling most efficiently with an air plasma. It would be of interest to probe these same parameters with other test gases like $\mathrm{CO}_{2}$ and $\mathrm{N}_{2}$ as well as various mixes of those two with $\mathrm{O}_{2}$, and such tests are planned for the future.

## C. Operational Envelope

Initial measurements of the torch included probing the operating envelope. For the initial test campaign, only air was used as a test gas. The torch has two user inputs: the DC voltage input to the RF oscillator circuit, $V_{a}$, and the test gas flow rate, $\dot{m}$. The two user inputs were varied independently to determine the conditions under which the torch could achieve stable operation.

The operational envelope for air is shown in Figure 6. The black points indicate the test conditions probed, and the gray region indicates the parameters under which stable operation was achieved. Below 9.5 kV , the torch is not producing the power necessary to sustain gas breakdown. The upper voltage bound is prescribed by the maximum allowable voltage for the components in the oscillator circuit. For each voltage, there is a maximum gas flow rate above which the torch cannot input enough power to sustain the ionization reactions and the plasma goes out. At the low end of flow rate, the coupling efficiency between the coil and the plasma drops and the temperature of the coolant increases above the allowable limits for operation.

## D. Plasma Power and Efficiency

The amount of energy deposited into the test gas by the inductor coil is estimated using the known power input into the torch (which is calculated from the anode voltage and anode current) and the amount of energy lost to the cooling system (which is calculated using the heat capacity of water, the coolant flow rate and the temperature change). It is assumed that the remainder of the energy goes into the plasma and is reported as the "bulk enthalpy" of the plasma. This value is more of an upper bound on the true enthalpy of the plasma as there are many more pathways for energy loss than just the cooling system.

The bulk enthalpy for an air plasma is plotted over the operating range of the torch in Figure 7. It can be seen that the bulk enthalpy decreases monotonically as the flow rate is increased. This is expected, as the voltage input is the primary driver of the plasma power. So for a constant voltage, the plasma power is approximately constant, therefore as the flow rate is increased, the bulk enthalpy decreases.

The opposite is true for the power efficiency plotted in Figure 8. All of the anode voltage settings follow the same curve that seems to asymptote to a value somewhere between $45 \%$ and $50 \%$ at the highest flow rates and drops to $25 \%$ at the lowest operable flow rate.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-07.jpg?height=727&width=1115&top_left_y=298&top_left_x=482)
Figure 7. Bulk enthalpy of air for the operating range of the torch using air as a test gas

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-07.jpg?height=730&width=1096&top_left_y=1203&top_left_x=501)
Figure 8. Power efficiency of the torch for the operating range of the torch using air as a test gas.

## E. Spectroscopic Thermometry

Figure 9(a,b) shows a sample of calibrated spectra of air and argon respectively. In the air plasma emission spectrum, the dominant visible lines are the 777.3 nm oxygen triplet and groupings of nitrogen lines at $823 \mathrm{~nm}, 843 \mathrm{~nm}$, and 868 nm .

The Boltzmann temperature of the air plasma is calculated from the emission spectrum using the number densities of atoms contributing to the 615 nm oxygen line and the 777.3 nm oxygen triplet. These two lines are used because they are the only two that are intense enough to be discerned from the noise with the spectrometry equipment used for these measurements. The number densities of each excited state are determined using the relation

$$
\begin{equation*}
I=n_{u} \frac{A_{u l}}{4 \pi}\left(\varepsilon_{u}-\varepsilon_{l}\right) \tag{2}
\end{equation*}
$$

where the subscripts $u$ and $l$ represent the upper and lower state of the transition, respectively.
The Boltzmann temperature can then be determined from the number densities in each excited state using the following relation.

$$
\begin{equation*}
\frac{n_{u 1}}{g_{u 1}}=\frac{n_{u 2}}{g_{u 2}} \exp \left(-\frac{\varepsilon_{u 1}-\varepsilon_{u 2}}{k T_{B}}\right) \tag{3}
\end{equation*}
$$

Substituting Eq. (2) into Eq. (3) equates the Boltzmann temperature to the intensity ratio between the two excited states, eliminating the need for an absolute intensity measurement. Applying this equation to the spectrum in Figure 9a yields a value of $T_{B}=6,350 \mathrm{~K}$.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-08.jpg?height=497&width=1557&top_left_y=829&top_left_x=292)
Figure 9. Sample air plasma (a) and argon plasma (b) emission spectra

However, using a raw spectral intensity measurement will lead to error in the measurement as the spectrometer is capturing light from all along its line of sight. This effect can be removed from the measurement by collecting emission spectra along several lines of sight at different radial locations, and then performing an Abel inversion. Using an assumption of axisymmetry of the plasma plume properties, the true radial distribution of spectral intensity can be back-calculated from the recorded line-of-sight emission spectra, and this true intensity can be used in Eq. (2) and Eq. (3) to calculate a plasma temperature as a function of plume radius.

In order for the assumption of axisymmetry to hold, the plume center must be accurately determined. Despite careful setup, the optics could only be aligned to the plume center within 1.5 mm . The offset of the coordinate system of the measurements from the true center of the plume was determined by finding the offset, $\delta$, which maximized the goodness of fit of a symmetric polynomial, $I_{p o l y}(x-\delta)$, to the data, $I_{c}(x ; \lambda)$, for every value of $\lambda$. The polynomial fit to the data, $I_{\text {poly }}(x-\delta ; \lambda)$, was then Abel inverted to calculate $I(\lambda ; r) . I_{c}, I_{\text {poly }}$, and the Abel inverted intensity, $I$, are all plotted in Figure 10 for the two oxygen emission lines.

In the calculation of temperature, the intensity value of each emission line was estimated by fitting a Gaussian distribution in $\lambda$ to the intensity data around the emission line for each value of $r$. Since instrument broadening, which is Gaussian in character, is the dominant cause of line broadening in these measurements, a Gaussian distribution provides the best fit to the data. The resulting $I_{777.3}(r)$ and $I_{615}(r)$ are then used in Eq. (2) and Eq. (3) to find $T(r)$.

Measurements were taken 5 mm downstream of the nozzle exit for three different torch settings: $10 \mathrm{kV}, 10.5 \mathrm{kV}$, and 11 kV , all at 25 slpm . The temperature profiles are shown in Figure 11. Note that the plume radius extends to 15 mm , but beyond 8 mm , the emission intensity was too weak to be discerned from the noise using the current collection system. The measurements show that for a flow rate of 25 slpm , increasing the voltage from 10 kV to 11 kV increases the peak temperature of the gas by approximately 750 K . For all three voltages, the temperature drops by 750 K to $1,200 \mathrm{~K}$ from the peak temperature at the centerline to the limit of the sensitivity of the collection optics at $r=8 \mathrm{~mm}$.

A fully developed temperature profile in a pipe has the form of a $4^{\text {th }}$ order polynomial in $r$. Since the measurement of the temperature profile was obtained very close to the nozzle exit, such a polynomial is a reasonable approximation to the temperature profile. A $4^{\text {th }}$ order polynomial fit to each temperature profile is also plotted in Figure 11 to show an approximate extrapolation of the temperature profile to the edge of the plume.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-09.jpg?height=858&width=1128&top_left_y=265&top_left_x=436)
Figure 10. Curve fit to intensity measurements and Abel-inverted intensity versus radius for $\lambda=777 \mathrm{~nm}$ and $\lambda=615 \mathrm{~nm}$.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-09.jpg?height=784&width=1063&top_left_y=1420&top_left_x=504)
Figure 11. Temperature vs radial location in the plume for $V_{a}=10 \mathrm{kV}, 10.5 \mathrm{kV}$, and 11 kV at $\dot{\boldsymbol{m}}_{\text {air }}=25 \mathrm{slpm}$. Dotted lines are $4^{\text {th }}$ order polynomial fits to the data points of the same color.

The excitation temperature of the plasma operating on argon was also determined. Because many more argon emission lines could be detected by the spectrometer setup than could be detected for air, the typically-used multipleline Boltzmann plot could be used, which tends to average out random error in the line intensity values.

The equation for the Boltzmann distribution takes the form of the relation

$$
\begin{equation*}
\ln \left(\frac{I \lambda}{g_{u} A_{u l}}\right)=-\frac{\varepsilon_{u}}{k_{B} T}+C \tag{4}
\end{equation*}
$$

Applying this equation to the spectroscopic data and measured intensity of the emission lines given in Table 1 and generating a linear fit produces the Boltzmann plot shown in Figure 12. From the linear fit, the temperature of the argon plasma was found to be $12,300 \mathrm{~K} \pm 1,220 \mathrm{~K}$, with the uncertainty calculated from the variation in multiple measurements.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-10.jpg?height=654&width=877&top_left_y=981&top_left_x=617)
Figure 12. Sample Boltzmann plot for argon spectrum

Table 1. Spectroscopic data for Ar emission lines used in temperature calculation
| $\lambda(\mathrm{nm})$ | $A\left(10^{6} \mathrm{~s}^{-1}\right)$ | $\varepsilon(\mathrm{eV})$ | $g_{u}$ |
| :---: | :---: | :---: | :---: |
| 675.2 | 1.9 | 14.743 | 5 |
| 687.1 | 2.78 | 14.710 | 3 |
| 696.5 | 6.40 | 13.320 | 3 |
| 703.0 | 2.67 | 14.840 | 5 |
| 714.0 | 0.625 | 13.282 | 3 |
| 763.5 | 24.5 | 13.172 | 5 |


![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-11.jpg?height=719&width=1472&top_left_y=336&top_left_x=336)
Figure 13. Stagnation point heat flux measurements from Gardon gauge.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-11.jpg?height=714&width=1459&top_left_y=1203&top_left_x=349)
Figure 14. Stagnation point heat flux for air measured with a slug calorimeter.

## F. Cold Wall Heat Flux Measurements

5. Stagnation Point Heat Flux

Gardon gauge measurements of the stagnation point cold wall heat flux with air as the test gas were taken at 1,2 , and 3 nozzle exit diameters downstream of the nozzle exit for the entire operating envelope of the torch. The results of this test campaign are shown in Figure 13 as a function of test gas flow rate. The points show individual measurements, and each line shows the average function of $\dot{m}_{\text {air }}$ for each voltage setting.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-12.jpg?height=720&width=909&top_left_y=262&top_left_x=271)
Figure 15. Selected comparison of Gardon gauge and slug calorimeter measurements at $\mathbf{Z}=\mathbf{6 0 ~ m m}$

Several interesting observations can be made from this data set. The heat flux peaks for a given anode voltage about halfway through the flow rate range, drops, and then levels off just before the maximum flow rate. This is likely due to the competing effects of the increasing flow velocity and decreasing bulk enthalpy as mass flow is increased for constant voltage. The nozzle exit velocity ranges from approximately $5 \mathrm{~m} / \mathrm{s}$ to $70 \mathrm{~m} / \mathrm{s}$ over the range of anode voltages and air flow rates. At air flow rates of above around 40 slpm , the torch plume also becomes very turbulent and this turbulent mixing with the room air might also help account for the drop in heat flux at high flow rates.

The curves for $V_{a}=11.9 \mathrm{kV}$ appear to be anomalous in that the heat flux for a given flow rate increases much more from 11.5 kV to 11.9 kV than it does from 11 kV to 11.5 kV . However, comparing the measurements to those taken with the slug calorimeter, plotted in Figure 14, the effect looks real. One explanation is that the plasma jet produced at $V_{a}=11.9 \mathrm{kV}$ maintains its hot core flow for longer before being dissipated by mixng with the room air. The measurements at $Z=90 \mathrm{~mm}$ lend credence to this hypothesis; the heat flux values for low flow rates are much lower than one might expect following the trend in heat flux vs distance from nozzle and then they suddenly jump up to the expected values at $\dot{m}=40 \mathrm{slpm}$, indicating that at 40 slpm , the jet becomes much more stable and is able to sustain for the full 90 mm from the nozzle to the probe.

Similar measurements were taken at the same Z values and a selected number of flow rates within the operating range using a slug calorimeter. These measurements are given in Figure 14. The change with mass flow rate for a constant voltage is qualitatively similar to that measured using the Gardon gauge. To make a better comparison between the two sets of data, representative sets of slug calorimeter and Gardon gauge measurements are plotted together in Figure 15 and show that the two data sets agree fairly well, though at higher heat flux values, the calorimeter seems to consistently read slightly higher.

## 6. Radial Variation in Heat Flux

Using the small-diameter Gardon gauge, measurements of the radial variation in heat flux were taken at several points in the operating range of the torch. Heat flux profiles for $V_{a}=9.5 \mathrm{kV}$ at three different flow rates are shown below in Figure 16. The first thing one might note is the peak of each curve is significantly larger than the values measured for the same voltage and flow rate by the large Gardon gauge and the slug calorimeter. However, since the large probes effectively average the heat flux readings over their entire area, one can average over the entire profile to approximate the reading of a larger diameter gauge. In Figure 18, the profiles for $V_{a}=9.5 \mathrm{kV}$ and $V_{a}=10 \mathrm{kV}$ have been averaged and are compared to the slug calorimeter measurements taken at the same conditions to show that the two are comparable.
The other point to note about Figure 16 is that the position of the maximum heat flux seems to drift away from the center as flow rate is increased. Figure 17 shows the peak heat flux offset for all of the radial profiles obtained and demonstrates that the effect is consistent across voltage settings and is more or less exclusively dependent on mass flow rate. The source of this asymmetry is not known at this time.

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-13.jpg?height=609&width=765&top_left_y=224&top_left_x=653)
Figure 16. Radial heat flux profiles for $\mathbf{V}_{\mathbf{a}} \boldsymbol{=} \mathbf{9 . 5 ~ k V}$

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-13.jpg?height=608&width=760&top_left_y=1043&top_left_x=276)
Figure 17. Offset of the position of peak heat flux with respect to the centerline of the nozzle

![](https://cdn.mathpix.com/cropped/2025_09_15_6f8f88d2b0faf7e1b809g-13.jpg?height=608&width=766&top_left_y=1043&top_left_x=1086)
Figure 18. Averaged heat flux profiles compared to measurements at the same voltage and flow rate taken by the slug calorimeter.

## IV. Conclusion

A new inductively coupled plasma torch facility at the University of Texas at Austin is characterized using emission spectroscopy and slug calorimetry. The torch can produce air and argon plasmas, or any combination of the two, with plasma bulk enthalpy of up to $40 \mathrm{MJ} / \mathrm{kg}$ and a mass flow rate of up to approximately 70 slpm of air or 85 slpm of argon. The plasma exits the discharge chamber through a 30 mm diameter subsonic nozzle.

Heat flux measurements using both a slug calorimeter and a Gardon gauge show a range of achievable heat fluxes between $80 \mathrm{~W} / \mathrm{cm}^{2}$ and $240 \mathrm{~W} / \mathrm{cm}^{2}$. The peak heat flux for a given anode voltage setting occurs at a flow rate of between 35 slpm and 45 slpm , most likely due to the interplay between the decreasing bulk enthalpy and increasing flow velocity with increasing flow rate. Some effects of plume stability as it interacts with the room air are also apparent in the heat flux data.

A small heat flux probe was used to measure the radial variation in heat flux and it was found that there is an asymmetry in the plume that increases with mass flow rate and is stable across multiple runs and multiple voltage settings.

Temperature measurements using emission spectroscopy have been performed using an OceanOptics HR4000 spectrometer. These measurements show a temperature in the air plasma of between $6,000 \mathrm{~K}$ and $8,000 \mathrm{~K}$ based on
the ratio of the intensity of the $777 \mathrm{~nm} \mathrm{O}_{2}$ triplet and the $615.7 \mathrm{~nm} \mathrm{O}_{2}$ line. Measurements of the argon plasma emission spectrum give a higher temperature of about $12,000 \mathrm{~K}$.

Radial temperature variations were also measured using an Abel inverted emission spectrum vs plume radius. Profiles were measured at $V_{a}=10 \mathrm{kV}$ and $V_{a}=11 \mathrm{kV}$ at the same flow rate of 25 slpm, and show a temperature difference of approximately 500 K between these two torch settings as well as a mean radial variation of approximately $180 \mathrm{~K} / \mathrm{mm}$ over the distance measured.

These measurements give a good baseline for designing future experiments that will include testing heat-shield material samples. Further work needs to be done in developing the capability to use test gases other than air and argon as well as determining the source of and possibly correcting the asymmetry in the plasma jet.

## Acknowledgments

This project is sponsored by NASA JSC under grant number NNX15AH17A.
Special thanks to Hai Nguyen at JSC for the design and assembly of the water-cooled insertion arm.

## References

[1] R. K. Crouch and G. D. Walberg, "An Investigation of Ablation Behavior of Avcoat 5026/39M Over a Wide Range of Thermal Environments," 1969.
[2] R. Savino, M. De Stefano Fumo, L. Silvestroni, and D. Sciti, "Arc-jet testing on HfB2 and HfC-based ultra-high temperature ceramic materials," J. Eur. Ceram. Soc., vol. 28, pp. 1899-1907, 2008.
[3] T. Ogasawara, T. Ishikawa, T. Yamada, R. Yokota, I. Masayoshi, and S. Nogi, "Thermal response and ablation characteristics of carbon fiber reinforced composite with novel silicon containing polymer MSP," J. Compos. Mater., vol. 36, no. 2, pp. 143-157, 2002.
[4] B. Laub, "Use of Arc-Jet Facilities in the Design and Development of Thermal Protection Systems," in 25th AIAA Aerodynamic Measurement Technology and Ground Testing Conference, 2006, no. 3292.
[5] T. B. Reed, "Induction-Coupled plasma torch," J. Appl. Phys., vol. 32, no. 5, pp. 821-824, 1961.
[6] B. Bottin, M. Carbonaro, O. Chazot, G. Degrez, D. Vanden Abeele, P. Barbante, S. Paris, V. Van Der Haegen, T. Magin, and M. Playez, "A decade of aerothermal plasma research at the von Karman institute," Contrib. to Plasma Phys., vol. 44, no. 5-6, pp. 472-477, 2004.
[7] M. Auweter-Kurtz, F. Hammer, G. Herdrich, H. Kurtz, T. Laux, E. Schreiber, and T. Wegmann, "The Ground Test Facilities for TPS at the Institut Für Raumfahrtsysteme," in Proceedings of the Third European Symposium on Aerothermodynamics for Space Vehicles, 1998.
[8] M. E. MacDonald, C. M. Jacobs, C. O. Laux, F. Zander, and R. G. Morgan, "Measurements of Air Plasma/Ablator Interactions in an Inductively Coupled Plasma Torch," J. Thermophys. Heat Transf., vol. 29, no. 1, pp. 12-23, 2014.
[9] W. Owens, J. Uhl, M. Dougherty, A. Lutz, J. Meyers, and D. Fletcher, "Development of a 30 kW Inductively Coupled Plasma Torch for Aerospace Material Testing," in 10th AIAA/ASME Joint Thermophysics and Heat Transfer Conference, 2010, no. 4322.
[10] A. Gutsol, J. Larjo, and R. Hernberg, "Comparative Calorimetric Study of ICP Generator with Forward-Vortex and Reverse-Vortex Stabilization," Plasma Chem. Plasma Process., vol. 22, no. 3, pp. 351-369, 2002.
[11] R. Villarreal and P. L. Varghese, "Frequency-resolved absorption tomography with tunable diode lasers," Appl. Opt., vol. 44, no. 31, pp. 6786-6795, 2005.


[^0]:    ${ }^{1}$ Graduate Research Assistant, Dept. of Aerospace Engineering and Engineering Mechanics, Mail Stop C0600, Member AIAA.
    ${ }^{2}$ Bob R. Dorsey Professor in Engineering, Dept. of Aerospace Engineering and Engineering Mechanics, Mail Stop C0600, Member AIAA
    ${ }^{3}$ Stanley P. Finch Centennial Professor in Engineering, Dept. of Aerospace Engineering and Engineering Mechanics, Mail Stop C0600, Member AIAA
    ${ }^{4}$ Project Manager, NASA Johnson Space Center.
    ${ }^{5}$ RHTF Test Director, Thermal Design Branch, NASA Johnson Space Center, Mail Stop ES3, Member AIAA.

