# The H9 Magnetically Shielded Hall Thruster 

IEPC-2017-232

Presented at the 35th International Electric Propulsion Conference<br>Georgia Institute of Technology • Atlanta, Georgia • USA<br>October 8-12, 2017

Richard R. Hofer, ${ }^{1 *}$ Sarah E. Cusson, ${ }^{2 \dagger}$ Robert B. Lobbia ${ }^{1 \ddagger}$ and Alec D. Gallimore ${ }^{2 \S}$<br>${ }^{1}$ Jet Propulsion Laboratory, California Institute of Technology, Pasadena, CA<br>${ }^{2}$ University of Michigan, Ann Arbor, MI


#### Abstract

We describe an overview of the design process and acceptance testing of the magnetically shielded H9, a $9-\mathrm{kW}$ class Hall thruster developed by the Jet Propulsion Laboratory in collaboration with the University of Michigan and the Air Force Research Laboratory. The new thruster continues the collaborative tradition of the unshielded H6 as a testbed for studies of thruster physics and developments in diagnostics and thruster technology. The H9 inherits the LaB ${ }_{6}$ cathode, anode/gas distributor, and discharge chamber geometry from the H6. New elements include the magnetic circuit, graphite pole covers, and the thermal and mechanical design. The magnetic circuit has been improved to achieve a higher degree of magnetic shielding at discharge voltages greater than 300 V , and consequently, the higher discharge power capability of 9 kW at up to 800 V . Three copies of the H9 have now completed fabrication, assembly, and acceptance testing. Total efficiency ranges from $61-63 \%$ over $300-800 \mathrm{~V}$ and specific impulse reaches 2950 s at $800 \mathrm{~V}, 9 \mathrm{~kW}$.


## I. Introduction

MAGNETIC shielding (MS) in Hall thrusters is a lifetime technology that effectively eliminates discharge chamber erosion as a wearout failure mechanism [1-4]. The physics of magnetic shielding were first identified through numerical simulations performed by the Jet Propulsion Laboratory (JPL) of the Aerojet Rocketdyne (AR) XR-5 Hall thruster [1], subsequently validated through simulations and experiments on the magnetically shielded H6MS Hall thruster [3-5], and then extended to high-specific impulse $[6,7]$, high-power $[8]$, low-power $[9,10]$, and conducting walls

[^0]Copyright 2017. All rights reserved.
[11,12]. Magnetic shielding has also been incorporated in the joint NASA Glenn Research Center (GRC) and JPL development of the 12.5 kW Hall Effect Rocket with Magnetic Shielding (HERMeS) [13]. With a design lifetime of 50 kh and operation up to 3000 s specific impulse, the application of HERMeS potentially enables multiple human and robotic missions [14-18].

Depicted schematically in Figure 1, magnetic shielding provides for long-life operation by exploiting two fundamental properties of the Hall thruster discharge: 1) the isothermality of magnetic field lines and 2) the thermalized potential, $\phi_{0} \approx \phi-T_{e 0} \times \ln \left(n_{e} / n_{e 0}\right)$, which is constant along a magnetic field line if the electrons are also isothermal. Here, $\phi$ is the plasma potential, $T_{e 0}$ is the electron temperature, $n_{e}$ is the plasma density, and $n_{e 0}$ is a reference plasma density usually chosen on channel centerline. Since the plasma density decreases towards the wall, the plasma potential must drop unless the electron temperature is negligibly small. Magnetic shielding seeks to achieve equipotentialization of the lines of force by shaping lines to extend deep in the channel where electrons are cold ( $T_{e}<5 \mathrm{eV}$ ), thereby minimizing the potential drop due to the electron pressure and sustaining potentials near the wall close to the discharge voltage. Further, since the sheath potential is a strong function of the electron temperature, the cold electrons reduce the sheath potential to just a few volts. With the plasma potential near the wall maintained at nearly the discharge voltage and the sheath voltage being small, the kinetic energy that ions gain through the potential drop in the plasma along surfaces can be reduced significantly and with it the wall erosion rate.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-02.jpg?height=630&width=1587&top_left_y=1176&top_left_x=292)
Figure 1 Schematics of the upper half of the annular discharge chamber in a magnetic-layer Hall thruster (top) and profiles of $\phi$ and $T_{e}$ (bottom) established during ion acceleration. Left: Basic features of the thruster and typical profiles along the channel centerline. Middle: Representative magnetic field lines and profiles along the wall in an unshielded configuration. Right: Representative magnetic field lines and profiles along the wall in a magnetically shielded configuration.

As the H6MS was a retrofit of its earlier, unshielded (US) self (the H6US), the magnetic circuit modifications were not fully optimized. This led to some limitations in the strength and shape of the magnetic field topography, but the thruster still proved to be more than adequate for the early work exploring the capabilities of magnetic shielding $[4-6,11,19-21]$. Without the constraints of a retrofit, magnetic shielding effective at high-voltage was achieved in the
subsequent HERMeS design, which was experimentally verified by the less than 5 eV electron temperatures measured at the wall during 800 V operation [22]. At 12.5 kW though, HERMeS stresses the limits of all but the most capable vacuum facilities in the United States, which reduces the diversity of research institutions that can contribute to furthering our understanding of magnetic shielding physics.

In order to further additional investigations in the United States, a new 9 kW Hall thruster with magnetic shielding has been designed by JPL in collaboration with the University of Michigan (UM) and the Air Force Research Laboratory (AFRL). The new thruster, designated H9MS or simply H9, retains several features from the H6MS, most importantly the channel dimensions, but without the constraints of the original retrofit so that a higher degree of magnetic shielding is obtained. This improved magnetic circuit enables full-shielding to be obtained at discharge voltages greater than 300 V , and consequently, the higher discharge power capability of 9 kW at up to 800 V . Additionally, because the design of HERMeS incorporates features from related thrusters such as the H6MS, NASA-173M, and NASA-300M [23], the H9 can also be described as a sub-scale cousin of HERMeS.

This paper describes an overview of the design process and acceptance testing of the three H9 thrusters that have been fabricated, assembled, and tested. A companion paper describes a detailed characterization study of the thrusters [24] and other works are already using the thruster as a platform for fundamental research investigations [25,26].

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-03.jpg?height=682&width=1589&top_left_y=1219&top_left_x=268)
Figure 2 Left - The H9 magnetically shielded Hall thruster, shown without a cathode. Right The $\mathbf{L a B}_{\mathbf{6}}$ hollow cathode that is mounted on thruster centerline in the $\mathbf{H 9}$.

## II. Design Process Overview

## A. Project Organization

Originally designed in 2006, the H6 is a nominally 6 kW laboratory Hall thruster that was developed as a testbed for studies of thruster physics and developments in diagnostics and thruster technology [27]. This version is referred to here as the H6US in order to indicate that the discharge channel is unshielded from ion erosion. With funding for the project provided by AFRL, the
thruster was a joint development between JPL, UM, and AFRL. Three copies of the thruster were fabricated and distributed to each institute. High-performance was achieved through the use of a plasma lens magnetic field topography [28-31], a centrally-mounted lanthanum hexaboride $\left(\mathrm{LaB}_{6}\right)$ cathode [32-35], and a high-uniformity gas distributor/anode assembly [36]. At the nominal 300 V, 6 kW condition, thrust, total specific impulse, and total efficiency are $401 \mathrm{mN}, 1950 \mathrm{~s}$, and $64 \%$, respectively. At $800 \mathrm{~V}, 6 \mathrm{~kW}$, thrust, total specific impulse, and total efficiency are 274 mN , 3170 s , and $70 \%$, respectively.

In 2011, JPL modified the magnetic circuit of the H6 to incorporate magnetic shielding using physics-based simulations of the plasma and magnetic circuit. The new configuration, dubbed the H6MS, was shown through experiments and simulations to decrease wall erosion by three orders of magnitude relative to the US variant, effectively eliminating channel erosion as a failure mode [2-5,11,37-39]. At the nominal $300 \mathrm{~V}, 6 \mathrm{~kW}$ condition, thrust, total specific impulse, and total efficiency of the H6MS are $384 \mathrm{mN}, 2000 \mathrm{~s}$, and $62 \%$, respectively [5]. At $800 \mathrm{~V}, 9 \mathrm{~kW}$, thrust, total specific impulse, and total efficiency are $390 \mathrm{mN}, 3020 \mathrm{~s}$, and $64 \%$, respectively.

After several years of testing at JPL of the H6MS, it became clear that a new thruster was needed that could be shared within the United States for collaborative research purposes. The new design would not be constrained by the limitations of the H6US retrofit, but would retain certain features, such as the channel geometry, so that previous research results on the H6US and H6MS could still be leveraged in comparative studies.

The H9 design was led by JPL in collaboration with UM and AFRL. Table 1 lists the major project phases during the design, fabrication, assembly, and acceptance testing of the H9. For a given activity, blue shading in the table indicates the lead partner. The major differences between the H6 and H9 are the implementation of magnetic shielding developed at JPL and the capability to operate at a nominal power of 9 kW at a discharge voltage greater than 600 V . Additional details of the thruster configuration and throttling capabilities are described in subsequent sections.

Table 1 Major project phases during the design, fabrication, assembly, and acceptance testing of the H9. Blue shading indicates the lead partner for a given activity.
|  | JPL | UM | AFRL |
| :--- | :--- | :--- | :--- |
| Design |  |  |  |
| Specification | $\checkmark$ |  |  |
| Magnetic Circuit | $\checkmark$ | $\checkmark$ |  |
| Thermal/Mechanical | $\checkmark$ | $\checkmark$ |  |
| Cathode | $\checkmark$ |  |  |
| Engineering Drawings | $\checkmark$ | $\checkmark$ |  |
| Fabrication \& Assembly |  |  |  |
| Thruster fabrication | $\checkmark$ | $\checkmark$ |  |
| Thruster assembly | $\checkmark$ | $\checkmark$ |  |
| Anode fabrication \& assembly |  |  | $\checkmark$ |
| Acceptance Testing Locations |  |  |  |
| Magnetic Circuit | $\checkmark$ |  |  |
| Gas flow uniformity | $\checkmark$ |  |  |
| Functional, Stability, Plume |  | $\checkmark$ |  |
| Thrust, Stability | $\checkmark$ |  |  |


The 35th International Electric Propulsion Conference, Georgia Institute of Technology, USA October 8-12, 2017

## B. Purpose and Objectives

The primary purpose for the H 9 was to continue the collaborative tradition of the H 6 as a testbed for studies of thruster physics and developments in diagnostics and thruster technology. The specific objectives of the new thruster were to:

- Implement a magnetic circuit that retains magnetic shielding at discharge voltages up to 800 V,
- Provide a medium power ( $5-10 \mathrm{~kW}$ ), high-specific impulse ( $2500-3000 \mathrm{~s}$ ) laboratory thruster that can be tested at less than $20 \mu$ Torr in a wide variety of vacuum facilities in the United States,
- Provide a mechanical design that is simple and flexible enough to accommodate various plasma diagnostics and potential hardware changes,
- Provide a common thruster with a well-documented configuration across several institutions so that results can be easily compared, and
- Serve as a benchmark for the inner channel of a two-channel Nested Hall Thruster (NHT) incorporating magnetic shielding. This new development, which is being led by UM in collaboration with JPL, will demonstrate the ability to magnetically shield NHTs and could potentially be integrated in to UM's 100 kW -class X3 NHT [40].


## C. Specification

The H9 design specification was derived from the motivation and objectives in the previous section. Similar to the NASA-173M design relative to the P5 [28], the H9 magnetic field topography is the primary design variable that was changed relative to the H6MS and H6US. In this way, we can compare the operation and performance of the H 9 relative to the H 6 while holding other design features essentially constant. The main elements of the H6MS that are retained include:

- The discharge chamber geometry (mid-channel diameter, channel width, channel length)
- A segmented discharge channel with replaceable rings near the exit region
- The anode/gas distributor $[36,41]$
- A compact, centrally-mounted $\mathrm{LaB}_{6}$ hollow cathode [ 32,33 ]

As elaborated on further in the sections that follow, new elements of the H9 design include:

- A magnetic circuit that provides a higher degree of magnetic shielding than the H6MS
- Improvements in the thermal and mechanical design that enable 9 kW operation
- Incorporation of graphite pole covers to mitigate pole erosion [42]

Figure 3 depicts the operating envelope of the H9 and several Reference Firing Conditions (RFCs), which are listed in Table 2, that are used to benchmark performance. Aside from operation demonstrated at the RFCs, the operating envelope is a conservative prediction based on experience operating the H6US, H6MS, and HERMeS [5,6,43]. For example, the upper current limit of 40 A is based on H6US operation [44,45], the lower current limit of 2 A is based on the H6MS [19], and the minimum current for voltages greater than 300 V is estimated from experience operating the H6MS and the "high-I $\mathrm{I}_{\mathrm{sp}}$ truncation" for HERMeS described in Ref. [43]. Operation up to 12 kW is an expected capability of this laboratory design, but is not included here as it may prove challenging if this laboratory model were developed for flight. The maximum, estimated thrust of 560 mN for this operating envelope occurs at $225 \mathrm{~V}, 40 \mathrm{~A}$. At 600 V , the maximum
specific impulse at the minimum power would be about 2600 s , while at 800 V , the maximum specific impulse at the minimum power would be about 2900 s .

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-06.jpg?height=609&width=868&top_left_y=403&top_left_x=664)
Figure 3 H9 operating envelope and Reference Firing Conditions used during the initial testing of each thruster.

Table 2 Reference Firing Conditions used during the initial testing of each H9 thruster.
| $\mathbf{V d}(\mathbf{V})$ | $\mathbf{I d}(\mathbf{A})$ | $\mathbf{P d}(\mathbf{k W})$ |
| :---: | :---: | :---: |
| 300 | 15 | 4.5 |
| 300 | 20 | 6.0 |
| 400 | 15 | 6.0 |
| 500 | 15 | 7.5 |
| 600 | 15 | 9.0 |
| 800 | 11.25 | 9.0 |


## D. Component Features

## 1. Discharge Chamber

Schematically illustrated in Figure 4, the boron nitride discharge chamber retains the geometry of the H6MS in terms of the mid-channel diameter, channel width, channel length, and the chamfer angle in the exit region.

The segmented channel is also retained from the H6MS, which divides the discharge chamber in to a U-shaped "wall" region and replaceable "rings" near the exit. The rings allow for the lowcost integration of wall-probes [4,22] or studies of alternate wall material interactions with the plasma [11,46].

## 2. Anode/Gas Distributor

Propellant is injected in to the discharge chamber with a gas distributor that also serves as the anode. The gas distribution method, which was previously described in Ref. [36,41], achieves high-uniformity in the azimuthal direction and contributes to efficient operation of the thruster.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-07.jpg?height=502&width=673&top_left_y=241&top_left_x=726)
Figure 4 Schematic illustrating the segmented discharge chamber geometry and notional magnetic field of the H9. Not to scale.

## 3. Cathode

Electrons are supplied to the discharge via a lanthanum hexaboride ( $\mathrm{LaB}_{6}$ ) hollow cathode mounted on thruster centerline. This configuration, sometimes referred to as "centrally-mounted" or "internally-mounted" has been shown to minimize the cathode coupling voltage, reduce divergence, increase efficiency, and minimize the effects of vacuum facility backpressure on the performance and discharge stability $[19,34,35,45]$. This cathode, which has been used in all of the H6 and H9 thrusters, has been operated up to 60 A during development and was previously described in Ref. [32,33].

## 4. Magnetic Circuit

The retrofit of the H6MS magnetic circuit was well suited for 300 V discharges, but saturation of the pole pieces led to some field shape distortion at high magnetic field strength. The field distortion was actually predicted to limit the shielding on the inner wall under 800 V operation, which was confirmed during a 113 h wear test [6]. While the H6MS circuit limitation actually proved to be a convenient demonstration of magnetic shielding physics, reducing or eliminating these limitations was an important goal of the new magnetic circuit design for the H9.

Infolytica's Magnet v7.5 was used to numerically design the magnetic circuit. Changes to the pole pieces were made to improve the field shape so that magnetic shielding could be maintained up to 800 V . Saturation was reduced in the circuit so that higher field strengths could be achieved. To simplify the circuit construction and improve the field symmetry, the discrete outer coils used in the H6MS were replaced with a single outer coil that encompasses the discharge chamber. This simplifies the circuit construction and also eliminates azimuthal variations of the magnetic field in the near-field of the thruster. The improved field symmetry and the centrally-mounted cathode provide for an essentially axisymmetric system, which also improves the accuracy of twodimensional plasma simulations. Finally, pole piece covers fabricated from graphite were added to protect these surfaces from ion erosion.

## 5. Other Features

Additional features of the thruster of note include:

- The addition of a rectangular thruster mount that serves also as a radiator to reject waste heat from the discharge and maintain moderate temperatures in the thruster,
- Integration of thermocouples throughout the thruster assembly for monitoring the health of components during testing, and
- Operation of the thruster in a "cathode-tied" electrical configuration wherein the thruster body is electrically connected to the cathode. This configuration, which was first proposed by JPL and demonstrated on the H6MS during development of HERMeS [43,47,48], serves to regulate the ion energy impacting the pole piece covers so that long-life operation is achieved. The configuration also eliminates a parasitic leakage path of electrons through the vacuum facility walls that is not present in flight.


## III. Acceptance Testing

After completing the new design, fabrication was initiated. AFRL was responsible for fabricating the anodes while UM performed the majority of the machining needed for the new thrusters. Cathodes from the H6 thrusters were re-used, so no new cathodes were fabricated by JPL.
Figure 5 shows the three thrusters that were assembled in this initial build of the H9. Thruster assembly was performed by UM at JPL. After the magnetic circuit was assembled, measurements of the magnetic field were taken. Additionally, gas flow uniformity measurements of the anodes were taken prior to the final discharge chamber assembly. More detail on the magnetic field and gas flow measurements is provided below.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-08.jpg?height=641&width=1134&top_left_y=1230&top_left_x=498)
Figure 5. Three H9 magnetically shielded Hall thrusters have been fabricated and tested. The thrusters are shown without their centrally-mounted $\mathbf{L a B}_{\mathbf{6}}$ hollow cathodes.

After assembly, acceptance testing of the thrusters was initiated. Each thruster was transported to UM for the first hot firings. A spare H6 cathode provided by JPL was used in all of the thruster firing data reported here (after delivery, AFRL and UM are now using other LaB ${ }_{6}$ cathodes from the H6).

Testing at UM included initial outgassing of each thruster and functional testing at each Reference Firing Condition (RFC) shown in Table 2. Functional testing at UM of each H9 included: 1) current-magnetic field maps to characterize plasma oscillations, and 2) plume characterization of the ion current density, ion energy, and ion species fractions. An issue with a
new thrust stand at UM did not allow for reliable measurements. Since the gas flow, magnetic field, and functional testing all showed a high degree of similarity of operation for each thruster, it was decided to transport one H9 back to JPL for performance measurements. At JPL, measurements of performance and stability were taken on H9 SN03, which confirmed that the H9 was performing as expected.

The reminder of this section provides additional information on the acceptance testing campaign. Detailed descriptions of the experimental apparatus and more extensive comparisons of each thruster are provided in our companion paper [24].

## A. Neutral flow uniformity

Depicted schematically in Figure 6, the uniformity of the propellant injection through each anode was characterized with measurements of the neutral xenon pressure using an ionization gauge. With the apparatus installed in the Owens chamber at JPL, the gas pressure was measured while rotating the discharge chamber/anode assembly. Similar methods have been used with other thrusters such as the XR-5, H6, and HERMeS [41,49,50]. Measurements were performed under background pressures of $2-10 \mu$ torr-Xe at xenon flow rates of 2 and $20 \mathrm{mg} / \mathrm{s}$. Additional detail may be found in Ref. [24].

Figure 7 shows the relative pressure variation around the channel of each H 9 thruster with 20 $\mathrm{mg} / \mathrm{s}$ of cold xenon flowing through the anode. A high-degree of uniformity was measured around the azimuth of the discharge channel. To quantify the degree of uniformity, the pressure deviation from the mean was calculated as

$$
\begin{equation*}
\frac{\delta P_{\max }}{\langle P\rangle}=\frac{P(\theta)_{\max }-P(\theta)_{\min }}{\langle P\rangle} \tag{1}
\end{equation*}
$$

where values less than $10 \%$ were required for each anode to pass. Table 3 lists the maximum pressure deviation of each anode with 2 and $20 \mathrm{mg} / \mathrm{s}$ of xenon flowing. At $20 \mathrm{mg} / \mathrm{s}$, the pressure deviation ranged from $3.4-6.5 \%$, while at $2 \mathrm{mg} / \mathrm{s}$ the range was $4.1-9.2 \%$.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-09.jpg?height=564&width=988&top_left_y=1664&top_left_x=593)
Figure 6 Schematic illustrating the method used to measure the gas flow uniformity around the azimuth of the discharge channel.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-10.jpg?height=584&width=836&top_left_y=292&top_left_x=634)
Figure 7 Relative pressure around the channel of each H 9 with $20 \mathrm{mg} / \mathrm{s}$ of cold xenon flowing through the anode/gas distributor.

Table 3 Maximum pressure deviation relative to the mean for each $\mathbf{H 9}$ thruster measured during cold flow testing at 2 and $20 \mathrm{mg} / \mathrm{s}$ through the anode/gas distributor.
| Thruster | $\boldsymbol{\delta} \mathbf{P}_{\text {max }} /\langle\mathbf{P}\rangle \cdot \mathbf{1 0 0 \%} \mathbf{2 ~ m g} / \mathrm{s}$ | $\boldsymbol{\delta} \mathbf{P}_{\max } /<\mathbf{P}>\cdot \mathbf{1 0 0} \boldsymbol{\%}$ <br> $\mathbf{2 0 ~ m g} / \mathrm{s}$ |
| :--- | :--- | :--- |
| SN01 | 9.2\% | 6.5\% |
| SN02 | 4.6\% | 3.6\% |
| SN03 | 4.1\% | 3.4\% |


## B. Magnetic Circuit

Acceptance testing of the magnetic circuit included two-dimensional maps of the magnetic field and field strength measurements. The maps, which measured the radial and axial components of the magnetic field in the discharge chamber and near-field of the thruster, were used to verify the magnetic field simulations used in the thruster design. Excellent agreement was found between the simulation and the measurements as well as from thruster to thruster.

Figure 8 shows the relative magnetic field intensity, measured at the location of the maximum, radial magnetic field on channel centerline ( $\mathrm{B}_{\mathrm{r}, \max }$ ) versus inner coil current while the inner and outer coils are energized at a fixed current ratio [28]. A linear circuit response is shown up to $20 \%$ higher than the nominal field strength ( $\mathrm{B}_{\mathrm{o}}$ ). The response of each thruster was very similar. Additional details on the magnetic circuit acceptance testing and observations supporting that magnetic shielding is achieved during thruster operation is provided in our companion paper [24].

## C. Plume Ion Current Density

Plume measurements of the ion current density were taken on each H9 thruster during functional testing at UM. Figure 9 shows the radial variation from thruster centerline of the ion current density as measured with a Faraday probe at an axial position from the exit of the H 9 of $\mathrm{z} / \mathrm{D}_{\mathrm{m}}=6.3$, where $\mathrm{D}_{\mathrm{m}}$ is the H9 mid-diameter of the discharge channel. Consistent with the neutral flow uniformity measurements, the plume ion current density is highly symmetric about
thruster centerline. The double-peak structure in the figure is also well aligned with $\mathrm{r} / \mathrm{R}_{\mathrm{m}}=1$, where $R_{m}$ is the mid-channel radius, which is an indication of a well-collimated ion beam. Additional Faraday probe measurements from the functional testing of each thruster are presented in Ref. [24].

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-11.jpg?height=605&width=834&top_left_y=504&top_left_x=674)
Figure 8 Relative magnetic field intensity for each H9 thruster versus inner coil current while the inner and outer coils are energized at a fixed current ratio.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-11.jpg?height=610&width=901&top_left_y=1347&top_left_x=653)
Figure 9 Ion current density measured in the plume versus radial position from thruster centerline over $300-600 \mathrm{~V}$ at 15 A for H9 SN03 during testing at UM. The axial distance is $\mathrm{z} / \mathrm{D}_{\mathrm{m}}=6.3$.

## D. Performance

After functional testing of each thruster at UM, SN03 was returned to JPL for performance measurements. Figure 10 shows the H9 SN03 operating at $800 \mathrm{~V}, 9 \mathrm{~kW}$ in the Owens Chamber at JPL. The image shows a collimated beam with the telltale "dark space" regions near the discharge channel walls that are one indication of effective magnetic shielding [4]. Performance
testing of SN03 at JPL included operation over the Reference Firing Conditions shown in Table 1. Telemetry from the performance measurements are shown in Table 4 and Table 5. The inverted-pendulum thrust stand used has been employed in prior investigations of the H6, XR-5, SPT-140, and HERMeS [19,51-53]. The uncertainty of the thrust and flow rate measurements is $1 \%$ each. No corrections were made for the effects of neutral ingestion due to the backpressure that ranged from $10.4-13.5 \mu$ torr-Xe. Prior testing on the H6MS has shown that the thrust is nearly invariant with backpressure and that the flow rate correction to vacuum from these pressures is about $1 \%$ [19].

Telemetry from the thruster shows that the magnet power ranges from $0.8-1.5 \%$ of the discharge power over the RFCs. The voltage with respect to vacuum facility ground of the centrally-mounted, $\mathrm{LaB}_{6}$ hollow cathode ranged from -9.8 to -13.2 V . Current collected by the thruster chassis in the cathode-tied electrical configuration $[43,47]$ was -0.63 to -0.92 A , where the negative sign indicates net electron current is being recycled from the chassis through the hollow cathode emitter.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-12.jpg?height=751&width=1001&top_left_y=981&top_left_x=563)
Figure 10 Operation at $800 \mathrm{~V}, 9 \mathrm{~kW}$ of H9 SN03 at JPL.

Table 4 Telemetry from performance testing of SN03 at JPL.
| Vd (V) | Id (A) | Pd (kW) | Anode (mg/s) | Cathode (mg/s) | Inner Coil (A) | Outer Coil (A) | Pmag/ Pd | Thrust (mN) | Total Isp (s) | Total Eff | Pressure ( $\mu$ Torr-Xe) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 300 | 20 | 6.00 | 17.88 | 1.25 | 4.5 | 2.97 | 1.1\% | 379.1 | 2020 | 0.619 | 13.5 |
| 300 | 15 | 4.50 | 14.18 | 0.99 | 4.5 | 2.97 | 1.5\% | 290.1 | 1950 | 0.607 | 11.0 |
| 400 | 15 | 6.00 | 14.86 | 1.04 | 4.5 | 2.97 | 1.2\% | 347.7 | 2230 | 0.626 | 11.2 |
| 500 | 15 | 7.50 | 15.28 | 1.07 | 4.5 | 2.97 | 1.0\% | 394.7 | 2460 | 0.629 | 11.6 |
| 600 | 15 | 9.00 | 15.47 | 1.08 | 4.5 | 2.97 | 0.8\% | 436.2 | 2690 | 0.634 | 11.9 |
| 800 | 11.25 | 9.00 | 12.71 | 0.89 | 5 | 3.3 | 1.1\% | 391.0 | 2950 | 0.621 | 10.4 |


Table 5 Additional telemetry from performance testing of SN03 at JPL.
| Vd (V) | Id (A) | Pd (kW) | Vcg (V) | Chassis Current (A) | Anode P2P (V) | Anode P2P/Vd | Anode P2P (A) | Anode RMS (A) | Anode P2P/Id | Anode RMS/Id | 1st largest Id freq (kHz) | 2nd largest Id freq (kHz) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 300 | 20 | 6.00 | -9.8 | -0.87 | 38 | 13\% | 6.4 | 0.99 | 32\% | 5\% | 9 | 76 |
| 300 | 15 | 4.50 | -11.4 | -0.43 | 24 | 8\% | 5.2 | 0.70 | 34\% | 5\% | 9 | 76 |
| 400 | 15 | 6.00 | -11.5 | -0.60 | 51 | 13\% | 11.0 | 1.34 | 73\% | 9\% | 18 | 64 |
| 500 | 15 | 7.50 | -11.5 | -0.63 | 59 | 12\% | 9.8 | 1.84 | 66\% | 12\% | 67 | 18 |
| 600 | 15 | 9.00 | -11.1 | -0.92 | 69 | 12\% | 10.2 | 2.30 | 68\% | 15\% | 64 | 15 |
| 800 | 11.25 | 9.00 | -13.2 | -0.68 | 74 | 9\% | 8.8 | 1.35 | 78\% | 12\% | 73 | 33 |


Figure 11 shows the thrust, total specific impulse, and total efficiency of the H9 SN03 during operation at JPL. Thrust ranged from 290 mN at $300 \mathrm{~V}, 15 \mathrm{~A}$ to 436 mN at $600 \mathrm{~V}, 15 \mathrm{~A}$. Total specific impulse ranged 1950 s at $300 \mathrm{~V}, 15 \mathrm{~A}$ to 2950 at $800 \mathrm{~V}, 11.25 \mathrm{~A}$. Total efficiency was greater than $60 \%$ at all operating conditions, reaching a maximum of $63.4 \%$ at $600 \mathrm{~V}, 15 \mathrm{~A}$.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-13.jpg?height=1079&width=1541&top_left_y=1084&top_left_x=284)
Figure 11 Performance of H9 SN03 at JPL.

## E. Plasma oscillations

Plasma oscillations were characterized during functional testing at UM on all three thrusters and during performance testing of SN03 at JPL. Table 5 lists the discharge voltage peak-to-peak
( P 2 P ), discharge current P 2 P and root-mean-square (RMS), and the first and second frequencies corresponding to the largest amplitude in the power spectral density of the discharge current. Figure 12 plots the P2P and dominant frequency data for the discharge current from Table 5. Figure 13 is the power spectral density of each RFC from SN03 during the JPL testing. Finally, Figure 14 plots the mean current and the P2P current versus magnetic field intensity for operation at $300 \mathrm{~V}, 15$ and $600 \mathrm{~V}, 15 \mathrm{~A}$ of SN03 during the functional testing at UM.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-14.jpg?height=546&width=1562&top_left_y=593&top_left_x=265)
Figure 12 Peak-to-peak discharge current oscillations (left) and dominant frequency of current oscillations (right) for H9 SN03 during operation at JPL.

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-14.jpg?height=830&width=1321&top_left_y=1355&top_left_x=382)
Figure 13 Power spectral density of the discharge current oscillations for each Reference Firing Condition of H9 SN03 during testing at JPL.

Overall, the plasma oscillation data show that the H9 discharge is stable over the RFCs, with maximum P2P current amplitudes of $78 \%$ of the mean and oscillation frequencies that are typical of magnetically shielded Hall thrusters [5,19-21,54]. The large jump in P2P oscillation magnitude is consistent with a mode transition in Hall thrusters that typically occurs in the range of 400-600 V. The dominant frequency also jumps from 18 to 67 kHz between 400 and 500 V as the mode transition completes from being dominated by the $10-30 \mathrm{kHz}$ breathing mode oscillations to one dominated by the $60-80 \mathrm{kHz}$ cathode oscillations first described by Jorns [20]. The variation of the current oscillations with magnetic field intensity also displays a high degree of magnetic stability margin [43] where the mean and P2P current are relatively invariant with magnetic field intensity. The amplitude and frequency content of discharge current oscillations is similar to H6MS and HERMeS [5,20,54-56].

![](https://cdn.mathpix.com/cropped/2025_09_22_287c121c88f9c7d6f646g-15.jpg?height=617&width=1560&top_left_y=853&top_left_x=265)
Figure 14 Mean discharge current and peak-to-peak current oscillations versus magnetic field intensity for H9 SN03 during constant flow rate operation at UM.

## IV. Conclusion

The design, fabrication, assembly, and testing of the H9 magnetically shielded Hall thruster has been completed. The thruster is an improvement over the H6MS, which was a retrofit of the unshielded H6. Key features are maintained from the H6MS, with the magnetic circuit being the primary change. Three new thrusters have now been fabricated and have all been fired in vacuum facilities at their final destinations at UM, AFRL, and JPL.

Characterization of the H9MS shows that high-performance is maintained over 300-800 V, with total efficiencies greater than $60 \%$ and specific impulse reaching 2950 s at $800 \mathrm{~V}, 9 \mathrm{~kW}$. The high-performance is attributed to the high-uniformity anode and centrally-mounted cathode, both used previously in the H6MS, and a magnetic field topography that retains key features necessary for efficient ion creation and acceleration while simultaneously providing for essentially erosion free operation of the discharge chamber walls.

The new thruster is intended to serve as a testbed for investigations of fundamental physics in magnetically shielded Hall thrusters in the United States. Institutions outside of those of this initial collaboration should contact JPL if they are interested in fabricating copies for their own research programs.

## Acknowledgments

Portions of the research described in this paper were carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration. The authors would like to acknowledge funding provided by the NASA Space Technology Research Fellowship grant number NNX15AQ43H.

## References

[1] Mikellides, I. G., Katz, I., Hofer, R. R., Goebel, D. M., De Grys, K. H., and Mathers, A., "Magnetic Shielding of the Acceleration Channel in a Long-Life Hall Thruster," Physics of Plasmas 18, 033501 (2011).
[2] Mikellides, I. G., Katz, I., Hofer, R. R., and Goebel, D. M., "Magnetic Shielding of Walls from the Unmagnetized Ion Beam in a Hall Thruster," Applied Physics Letters 102, 2, 023509 (2013).
[3] Mikellides, I. G., Katz, I., Hofer, R. R., and Goebel, D. M., "Magnetic Shielding of a Laboratory Hall Thruster Part I: Theory and Validation," Journal of Applied Physics 115, 043303 (2014).
[4] Hofer, R. R., Goebel, D. M., Mikellides, I. G., and Katz, I., "Magnetic Shielding of a Laboratory Hall Thruster Part II: Experiments," Journal of Applied Physics 115, 043303 (2014).
[5] Hofer, R. R., Goebel, D. M., Mikellides, I. G., and Katz, I., "Design of a Laboratory Hall Thruster with Magnetically Shielded Channel Walls, Phase II: Experiments," AIAA-2012-3788, July 2012.
[6] Hofer, R. R., Jorns, B. A., Polk, J. E., Mikellides, I. G., and Snyder, J. S., "Wear Test of a Magnetically Shielded Hall Thruster at 3000 Seconds Specific Impulse," Presented at the 33rd International Electric Propulsion Conference, IEPC-2013-033, Washington, DC, Oct 6-10, 2013.
[7] Mikellides, I. G., Hofer, R. R., Katz, I., and Goebel, D. M., "Magnetic Shielding of Hall Thrusters at High Discharge Voltages," Journal of Applied Physics 116, 5, 053302 (2014).
[8] Kamhawi, H., Huang, W., Haag, T., Shastry, R., Soulas, G., Smith, T., Mikellides, I. G., and Hofer, R. R., "Performance and Thermal Characterization of the NASA- 300 ms 20 kW Hall Effect Thruster," Presented at the 33rd International Electric Propulsion Conference, IEPC-2013-444, Washington, DC, Oct 6-10, 2013.
[9] Conversano, R. W., Goebel, D. M., Hofer, R. R., Matlock, T. S., and Wirz, R. E., "Development and Initial Testing of a Magnetically Shielded Miniature Hall Thruster," IEEE Transactions on Plasma Science 43, 1, 103-117 (2015).
[10] Grimaud, L. and Mazouffre, S., "Ion Behavior in Low-Power Magnetically Shielded and Unshielded Hall Thrusters," Plasma Sources Science and Technology 26, 5 (2017).
[11] Goebel, D. M., Hofer, R. R., Mikellides, I. G., Katz, I., Polk, J. E., and Dotson, B. N., "Conducting Wall Hall Thrusters," IEEE Transactions on Plasma Science 43, 1, 118-126 (2015).
[12] Grimaud, L. and Mazouffre, S., "Conducting Wall Hall Thrusters in Magnetic Shielding and Standard Configurations," Journal of Applied Physics 122, 3 (2017).
[13] Hofer, R., Polk, J., Mikellides, I., Lopez Ortega, A., Conversano, R., Chaplin, V., Lobbia, R., Goebel, D., Kamhawi, H., Verhey, T., Williams, G., Mackey, J., Huang, W., Yim, J. et al., "Development Status of the 12.5 kW Hall Effect Rocket with Magnetic Shielding (HERMeS)," Presented at the $35{ }^{\text {th }}$ International Electric Propulsion Conference, IEPC-2017-231, Atlanta, GA, Oct. 8 12, 2017, 2017.
[14] Strange, N., Landau, D., Polk, J., Brophy, J., and Mueller, J., "Solar Electric Propulsion for a Flexible Path of Human Space Exploration," Presented at the 61st International Astronautical Congress, IAC-10.A5.2.4, Prague, Czech Republic, Sep.-Oct., 2010.
[15] Strange, N., Merrill, R., Landau, D., Drake, B., Brophy, J., and Hofer, R., "Human Missions to Phobos and Deimos Using Combined Chemical and Solar Electric Propulsion," AIAA Paper 2011-5663, July 2011.
[16] Strange, N., Landau, D., Mcelrath, T., Lantoine, G., Lam, T., Mcguire, M., Burke, L., Martini, M., and Dankanich, J., "Overview of Mission Design for NASA Asteroid Redirect Robotic Mission Concept," Presented at the 33rd International Electric Propulsion Conference, IEPC Paper 2013-321, Washington, DC, Oct 6-10, 2013.
[17] Mcguire, M. L., Hack, K. J., Manzella, D. H., and Herman, D. A., "Concept Designs for NASA's Solar Electric Propulsion Technology Demonstration Mission," AIAA Paper 2014-3717, July 2014.
[18] Snyder, J. S., Manzella, D., Lisman, D., Lock, R. E., Nicholas, A., and Woolley, R., "Additional Mission Applications for NASA's 13.3-kW Ion Propulsion System," Presented at the IEEE Aerospace Conference, Big Sky, MT, March, 2016.
[19] Hofer, R. R. and Anderson, J. R., "Finite Pressure Effects in Magnetically Shielded Hall Thrusters," AIAA Paper 2014-3709, July 2014.
[20] Jorns, B. A. and Hofer, R. R., "Plasma Oscillations in a $6-\mathrm{kW}$ Magnetically Shielded Hall Thruster," Physics of Plasmas 21, 5, 053512 (2014).
[21] Sekerak, M. J., Longmier, B. W., Gallimore, A. D., Brown, D. L., Hofer, R. R., and Polk, J. E., "Azimuthal Spoke Propagation in Hall Effect Thrusters," IEEE Transactions on Plasma Science 43, 1, 72-85 (2015).
[22] Shastry, R., Huang, W., and Kamhawi, H., "Near-Surface Plasma Characterization of the $12.5-\mathrm{kW}$ NASA TDU1 Hall Thruster," AIAA-2015-3919, July 2015.
[23] Hofer, R. R., Kamhawi, H., Mikellides, I., Herman, D., Polk, J., Huang, W., Yim, J., Myers, J., and Shastry, R., "Design Methodology and Scaling of the 12.5 kW HERMeS Hall Thruster for the Solar Electric Propulsion Technology Demonstration Mission," Presented at the 62nd JANNAF Propulsion Meeting, JANNAF-2015-3946, Nashville, TN, June 1-5, 2015.
[24] Cusson, S. E., Hofer, R. R., Lobbia, R. B., Jorns, B. A., and Gallimore, A. D., "Performance of the H9 Magnetically Shielded Hall Thrusters," Presented at the 35th International Electric Propulsion Conference, IEPC-2017-239, Atlanta, GA, Oct. 8-12, 2017.
[25] Brown, Z. A. and Jorns, B. A., "Dispersion Relation Measurements of Ion-Acoustic-Like Waves in the near-Field Plume of a 9kW Magnetically Shielded Hall Thruster," Presented at the 35th International Electric Propulsion Conference, IEPC-2017-387, Atlanta, GA, Oct. 8-12, 2017.
[26] Durot, C. J., Jorns, B. A., and Gallimore, A. D., "Laser-Induced Fluorescence Measurements in the Acceleration Region of a 9kW Magnetically Shielded Hall Thruster," Presented at the 35th International Electric Propulsion Conference, IEPC-2017-029, Atlanta, GA, Oct. 8-12, 2017.
[27] Haas, J. M., Hofer, R. R., Brown, D. L., Reid, B. M., and Gallimore, A. D., "Design of the H6 Hall Thruster for High Thrust/Power Investigation," Presented at the 54th JANNAF Propulsion Meeting, Denver, CO, May 14-17, 2007.
[28] Hofer, R. R., "Development and Characterization of High-Efficiency, High-Specific Impulse Xenon Hall Thrusters," Ph.D. Dissertation, Aerospace Engineering, University of Michigan, Ann Arbor, MI, 2004. (Also NASA/CR-2004-213099)
[29] Hofer, R. R., Jankovsky, R. S., and Gallimore, A. D., "High-Specific Impulse Hall Thrusters, Part 1: Influence of Current Density and Magnetic Field," Journal of Propulsion and Power 22, 4, 721-731 (2006).
[30] Hofer, R. R. and Gallimore, A. D., "High-Specific Impulse Hall Thrusters, Part 2: Efficiency Analysis," Journal of Propulsion and Power 22, 4, 732-740 (2006).
[31] Hofer, R. R., "Magnetically-Conformed, Variable Area Discharge Chamber for Hall Thruster, and Method," United States Patent No. 8,407,979 (April 2, 2013).
[32] Hofer, R. R., Goebel, D. M., and Watkins, R. M., "Compact LaB6 Hollow Cathode for a 6 kW Laboratory Hall Thruster," Presented at the 54th JANNAF Propulsion Meeting, Denver, CO, May 14-17, 2007.
[33] Hofer, R. R., Goebel, D. M., and Watkins, R. M., "Compact High-Current Rare-Earth Emitter Hollow Cathode for Hall Effect Thrusters," United States Patent No. 8,143,788 (Mar. 27, 2012).
[34] Hofer, R. R., Johnson, L. K., Goebel, D. M., and Wirz, R. E., "Effects of Internally-Mounted Cathodes on Hall Thruster Plume Properties," IEEE Transactions on Plasma Science 36, 5, 2004-2014 (2008).
[35] Goebel, D. M., Jameson, K. K., and Hofer, R. R., "Hall Thruster Cathode Flow Impact on Coupling Voltage and Cathode Life," Journal of Propulsion and Power 28, 2, 355-363 (2012).
[36] Reid, B. M., Gallimore, A. D., Hofer, R. R., Li, Y., and Haas, J. M., "Anode Design and Verification for a 6 -kW Hall Thruster," JANNAF Journal of Propulsion and Energetics 3, 1, 29-43 (2010).
[37] Mikellides, I. G., Katz, I., and Hofer, R. R., "Design of a Laboratory Hall Thruster with Magnetically Shielded Channel Walls, Phase I: Numerical Simulations," AIAA Paper 2011-5809, July 2011.
[38] Mikellides, I. G., Katz, I., Hofer, R. R., and Goebel, D. M., "Design of a Laboratory Hall Thruster with Magnetically Shielded Channel Walls, Phase III: Comparison of Theory with Experiment," AIAA-2012-3789, July 2012.
[39] Goebel, D. M., Hofer, R. R., and Mikellides, I. G., "Metallic Wall Hall Thrusters," United States Patent No. 9,453,502 (Sep. 27, 2016).
[40] Hall, S. J., Florenz, R. E., Gallimore, A., Kamhawi, H., Brown, D. L., Polk, J. E., Goebel, D. M., and Hofer, R. R., "Implementation and Initial Validation of a $100-\mathrm{kW}$ Class Nested-Channel Hall Thruster," AIAA Paper 2014-3815, July 2014.
[41] Reid, B. M., "The Influence of Neutral Flow Rate in the Operation of Hall Thrusters," Ph.D. Dissertation, Aerospace Engineering, University of Michigan, Ann Arbor, MI, 2009.
[42] Jorns, B. A., Dodson, C., Anderson, J., Goebel, D. M., Hofer, R. R., Sekerak, M., Lopez Ortega, A., and Mikellides, I., "Mechanisms for Pole Piece Erosion in a $6-\mathrm{kW}$ Magnetically-Shielded Hall Thruster," AIAA-2016-4839, July 2016.
[43] Hofer, R., Polk, J., Sekerak, M., Mikellides, I., Kamhawi, H., Verhey, T., Herman, D., and Williams, G., "The 12.5 kW Hall Effect Rocket with Magnetic Shielding (HERMeS) for the Asteroid Redirect Robotic Mission," AIAA-2016-4825, July 2016.
[44] Shastry, R., Hofer, R. R., Reid, B. M., and Gallimore, A. D., "Method for Analyzing ExB Probe Spectra from Hall Thruster Plumes," Review of Scientific Instruments 80, 063502 (2009).
[45] Jameson, K. K., "Investigation of Hollow Cathode Effects on Total Thruster Efficiency in a 6 kW Hall Thruster," Ph.D. Dissertation, Aerospace Engineering, University of California, Los Angeles, Los Angeles, 2008.
[46] Kamhawi, H., Gilland, J. H., Williams, G., Mackey, J., Huang, W., Haag, T., and Herman, D., "Performance and Stability Characterization of the HERMeS Thruster with Boron Nitride Silica Composite Discharge Channel," Presented at the $35^{\text {th }}$ International Electric Propulsion Conference, IEPC-2017-392, Atlanta, GA, Oct. 8-12, 2017, 2017.
[47] Katz, I., Lopez Ortega, A., Goebel, D. M., Sekerak, M. J., Hofer, R. R., Jorns, B. A., and Brophy, J. R., "Effect of Solar Array Plume Interactions on Hall Thruster Cathode Common Potentials," Presented at the 14th Spacecraft Charging Technology Conference, ESA/ESTEC, Noordwijk, NL, April 4-8, 2016.
[48] Hofer, R. R., Jorns, B. A., Brophy, J. R., and Katz, I., "Hall Effect Thruster Electrical Configuration," California Institute of Technology, USPTO patent application submitted, March 302017.
[49] Kamhawi, H., Huang, W., Haag, T., Yim, J., Chang, L., Clayman, L., Herman, D. A., Shastry, R., Thomas, R., Griffith, C., Myers, J., Williams, G., Mikellides, I. G., Hofer, R. R. et al., "Overview of the Development of the Solar Electric Propulsion Technology Demonstration Mission 12.5-kW Hall Thruster," AIAA Paper 2014-3898, July 2014.
[50] De Grys, K. H., Meckel, N., Callis, G., Greisen, D., Hoskins, A., King, D., Wilson, F., Werthman, L., and Khayms, V., "Development and Testing of a 4500 Watt Flight Type Hall Thruster and Cathode," Presented at the 27th International Electric Propulsion Conference, IEPC-2001-011, Pasadena, CA, Oct. 15-19, 2001.
[51] Conversano, R. W., Hofer, R. R., Sekerak, M. J., Kamhawi, H., and Peterson, P. Y., "Performance Comparison of the 12.5 kW HERMeS Hall Thruster Technology Demonstration Units," AIAA-2016-4827, July 2016.
[52] Garner, C. E., Jorns, B. A., Van Derventer, S., Hofer, R. R., Rickard, R., Liang, R., and Delgado, J., "Low-Power Operation and Plasma Characterization of a Qualification Model SPT-140 Hall Thruster for NASA Science Missions," AIAA-2015-3720, July 2015.
[53] Hofer, R. R., Goebel, D. M., Snyder, J. S., and Sandler, I., "BPT-4000 Hall Thruster Extended Power Throttling Range Characterization for NASA Science Missions," Presented at the 31st International Electric Propulsion Conference, IEPC-2009-085, Ann Arbor, MI, Sept. 20-24, 2009.
[54] Huang, W., Kamhawi, H., and Haag, T. W., "Plasma Oscillation Characterization of Nasa's HERMeS Hall Thruster Via High Speed Imaging," AIAA-2016-4829, July 2016.
[55] Kamhawi, H., Huang, W., Haag, T., Yim, J., Herman, D., Peterson, P. Y., Williams, G., Gilland, J., Hofer, R., and Mikellides, I., "Performance, Facility Pressure Effects, and Stability Characterization Tests of NASA's Hall Effect Rocket with Mangetic Shielding Thruster," AIAA-2016-4826, July 2016.
[56] Kamhawi, H., Huang, W., Shastry, R., Haag, T., Yim, J., Thomas, R., Herman, D., Myers, J., Williams, G., Hofer, R., Mikellides, I., and Sekerak, M., "Performance and Stability Characterization Test of Nasa's 12.5 kW Hall Effect Rocket with Magnetic Shielding " Presented at the 62nd JANNAF Propulsion Meeting, Nashville, TN, June 1-5, 2015.


[^0]:    * Supervisor, Electric Propulsion Group
    ${ }^{\dagger}$ Ph.D. Candidate, Department of Aerospace Engineering
    ${ }^{\ddagger}$ Staff Engineer, Electric Propulsion Group, formerly with the Air Force Research Laboratory, ERC Inc., Edwards AFB, CA
    ${ }^{§}$ Robert J. Vlasic Dean of Engineering, Richard F. and Eleanor A. Towner Professor of Engineering, and Arthur F. Thurnau Professor of Aerospace Engineering

