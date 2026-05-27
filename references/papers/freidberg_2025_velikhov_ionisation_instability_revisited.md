# The Velikhov-ionisation instability revisited: a new opportunity for MHD energy conversion?

- Author: Jeffrey Freidberg
- Affiliation: Plasma Science and Fusion Center, MIT, Cambridge, MA, 02139, USA
- Published online: 04 August 2025
- Journal: Journal of Plasma Physics, Volume 91, Issue 4, 2025/08
- DOI: [10.1017/S0022377825100482](https://doi.org/10.1017/S0022377825100482)
- Source PDF: `/Users/yiruxiao/Desktop/mathpix_yiru/data/inbox/the-velikhov-ionisation-instability-revisited-a-new-opportunity-for-mhd-energy-conversion.pdf`
- Source HTML: `/Users/yiruxiao/Desktop/mathpix_yiru/data/output/the-velikhov-ionisation-instability-revisited-a-new-opportunity-for-mhd-energy-conversion/publisher_html/source.html`

## Abstract

The work presented here revisits the Velikhov-ionisation instability, an instability first discovered in the early 1960s (Velikhov, E. P. 1962 1st International Conference on MHD Electrical Power Generation, Newcastle upon Tyne, England, p. 135). This mode strongly deteriorates the performance of magnetohydrodynamic (MHD) energy convertors in which the seed gas must be at a substantially higher temperature than the high density primary gas, the latter gas carrying almost all the energy. Specifically, a finite temperature difference is necessary for the MHD generator to successfully act as a topping cycle for nuclear (fission and fusion) power plants. The ionisation instability has thus been viewed for many years as a show stopper for MHD nuclear topping cycles. Even so, some experimental observations, never fully exploited, show that nearly full ionisation of the seed gas can stabilise this dangerous instability. One goal of the research presented here is to provide a first-principles theoretical explanation for these experimental observations. The stabilisation can theoretically produce high temperature ratios, of the order of 10, by carefully choosing the density of the unionised seed gas. A second goal of the research is to investigate whether or not the recent development of high-field, high-temperature REBCO (rare-earth barium copper oxide) superconductors can lead to substantially improved power plant efficiency. Here, it is shown that the answer is subtle – no clear conclusions can be drawn, a consequence of the fact that the new stability criterion is a local one. What is needed to assess overall plant efficiency is a global analysis. Additional work has recently been completed on a newly developed global model which answers this question and will be reported on in a future paper.

## Keywords

plasma instabilities, plasma devices

## 1. Introduction

The research presented here re-examines the Velikhov-ionisation instability, an instability first discovered theoretically and experimentally in the early 1960s (Velikhov 1962; Velikhov & Dykhne 1963; Velikhov, Dykhne & Shipuk 1965). This instability played a major role in the experimentally observed poor performance of magnetohydrodynamic MHD energy convertors, intended for use as topping cycles for nuclear (fusion and fission) power plants. The present research revisits this instability and suggests a possible cure. Since MHD energy conversion research has been dormant for nearly 30 years in the USA, the discussion begins with a brief historical review to introduce readers, both young and old, to the basic concepts. This should help put the present work in context, after which, the new contributions are described in more detail.

### 1.1. Review of MHD energy conversion

The idea of using MHD energy conversion in the overall design of electric power plants has been known for many years (see for instance Karlovitz 1940; Sporn & Kantrowitz 1959; Rosa 1987; Messerle 1994). One important application of MHD energy conversion is as a topping cycle for both fossil (coal and natural gas) and nuclear (fission and fusion) power sources. In simple terms, the main goal is to increase the total plant efficiency from approximately 35–40 % to 55–60 %, which would represent an enormous gain in fuel efficiency and economics.

Fossil fuel MHD generators are usually designed as open cycle systems. Here, the output furnace gas, denoted as the primary gas, consists of multiple species (e.g. $\mathrm{CO},\,\mathrm{CO}_{2},\,\mathrm{NO},\mathrm{etc}.$ for coal) plus a small added amount of a low ionisation potential seed gas (e.g. K). These are combined to make an electrically conducting gas, i.e. a weakly ionised plasma. The complex molecular primary gas flows directly from the power source through the MHD generator (the topping cycle), and then into a standard electricity producing steam generator (the bottoming cycle). It is ultimately vented to the atmosphere with only the small amount of seed gas recirculated. The direct flow from power source to atmospheric venting is the reason for the name ‘open cycle’. In an open cycle system, the primary gas and seed electron temperatures are approximately equal and relatively high, of the order of 2000–2500 K (see for instance Rosa (1987)).

However, nuclear fuel powered MHD generators, because of science and engineering constraints, typically must operate at lower temperatures, approximately 1000–1300 K maximum (NEA 2022; Sorbom *et al.* 2015). To overcome the low temperatures, they are designed as closed cycle systems, an idea suggested by Kerrebrock (1964), Kerrebrock & Hoffman (1964) and Sheindlin, Batenin & Asinovsky (1964). For a closed cycle system, the power source gas coolant, typically He, is passed through a heat exchanger in which the secondary gas is monatomic, typically Ar. Argon has the desirable property of a relatively long energy equilibration time between the seed electrons and the primary gas. This allows the preferentially Ohmically heated seed electrons to reach a higher temperature (e.g. 3000–5000 K) than the primary gas, and then maintain this higher temperature flowing along the MHD generator because of the long energy equilibration time and continual Ohmic heating. The higher electron temperature produces a good quality electrically conducting gas even though the primary gas is much cooler. Both the primary gas (e.g. Ar) and seed gas (e.g. K) are recirculated for economic reasons; hence, the name ‘closed cycle’.

In spite of these potentially attractive applications, an examination of the world’s existing industrial scale power plants shows that MHD energy conversion plays essentially no role. The question then is ‘What went wrong?’ For fossil fuel applications, MHD was in competition with combined cycle gas. MHD worked but not well enough in terms of performance and cost. Combined cycle gas won the competition – hence, no need for MHD, nor for that matter, coal.

Nuclear fission was not very popular in the 1970s–1990s. Even so, when closed cycle MHD experiments were built, they exhibited strong performance deterioration due to the Velikhov-ionisation instability. For many years, this instability was viewed as a show stopper for closed cycle nuclear applications. New sophisticated stabilisation methods, using radio frequency (RF) fields, were developed, but in terms of industrial applications, they were probably too late (Murakami, Okuno & Yamasaki 2005).

Consequently, with no obvious remaining attractive applications, the large worldwide MHD energy conversion programme was strongly curtailed in the 1990s. At present, perhaps the largest remaining closed cycle, MHD energy conversion programme for power applications exists in Japan. This is a highly regarded, modest in size (relative to fusion) programme located at the Okuno Laboratory, Tokyo Institute of Technology.

### 1.2. Revisiting MHD energy conversion – societal reason

With this as background, one can now ask why it makes sense to revisit the Velikhov-ionisation instability. There are three reasons, one societal, the other two, technical. The societal reason is as follows. At present, as compared with the 1990s, there is a much greater concern about climate change, particularly the production of $\mathrm{CO}_{2}$. Combined cycle gas, which produces approximately one half the $\mathrm{CO}_{2}/\text{Watt}$ as a coal plant, still generates enormous quantities of $\mathrm{CO}_{2}$. If the world is going to phase out natural gas in favour of power sources that produce no $\mathrm{CO}_{2}$, then nuclear fission, while still not highly popular but becoming more acceptable, and nuclear fusion, highly popular but not here yet, represent sources of baseload electricity satisfying this goal. Because of the high capital cost of either fission or fusion power plants, the efficiency gains resulting from an effective MHD topping cycle would be a big win.

### 1.3. Revisiting MHD energy conversion – stabilising the ionisation instability

Even though the societal demand for nuclear power may have increased, how can the very serious problem of the ionisation instability that plagued early experiments be overcome? The focus is on two ideas. The first is based on some experimental observations, never fully exploited by the power industry, which show that nearly full ionisation of the seed gas can stabilise this dangerous instability (Petit, Caressa & Valensi 1968; Petit & Valensi 1969; Hatori & Shioda 1974; Petit & Geffray 2008). It thus makes sense to try and derive a first-principles theory that explains why nearly full ionisation of the seed gas stabilises the mode. Of specific interest is learning how to practically produce this stabilisation in the MHD operational regime of interest.

### 1.4. Revisiting MHD energy conversion – access to high magnetic fields

Second, there has been a recent major technical advance that promises performance improvement. This is the development of industrial scale high-field, high-temperature REBCO (rare-earth barium copper oxide) superconducting magnets (Vieira *et al.* 2024). These magnets should produce magnetic fields of the order of 15–20 T as compared with early experiments which typically operated at 3–4 T. Intuitively, one might expect access to higher magnetic fields to improve the performance of applications based on MHD.

However, a more careful examination of the ionisation instability shows that the actual situation is much subtler when it comes to high field. To begin, note that the instability occurs when the temperature difference between the primary gas and seed electrons becomes too large. An analysis of the MHD equations shows that the stability boundary for a Hall MHD convertor, which is the primary interest as discussed later, recasts the temperature difference into to a maximum limit on the Hall parameter, defined as $\beta =\Omega _{e}/\nu _{M}=(eB_{0}/m_{e})/\nu _{M}\propto B_{0}$. Here, $\Omega _{e}$ is the electron cyclotron frequency and $\nu _{M}$ is the electron–primary gas momentum exchange collision frequency. The stability boundary sets a limit on the maximum allowable magnetic field: $\beta \leq \beta _{crit}$. Consequently, having access to higher fields above this limit would not appear to be of much use. It is of basic interest to learn which effect dominates – improved performance due to higher fields versus the high-field limit due to the instability?

### 1.5. Main contributions of the present work

The issues just discussed are analysed in the present paper. One main contribution is a derivation of a first-principles theory (i.e. revisiting the Velikhov-ionisation instability) that explicitly shows the connection between the maximum $\beta$ limit and the degree of ionisation (6.23). Note that a few theories have been developed that show that full ionisation is stabilising (Petit & Valensi 1969; Mitchner & Kruger 1973; Nakamura & Riedmuller 1974; Kien 2016). However, these theories do not fully exploit the ionisation stability criterion to show how high field could help or hurt the design of a Hall generator. A second main contribution uses the stability relation to show, in practical terms, that there is a critical maximum seed density for marginal stability, including the effects of higher magnetic fields.

Our results are then used to determine the overall impact of magnetic field on performance by the introduction of two simple intuitive figures of merit. The first is $S_{\Omega }/S_{L}$, where $S_{\Omega }$ is the local Ohmic heating power density and $S_{L}$ is the power density delivered to the load. One wants this ratio to be small – a certain fraction of the input power is converted to electricity and for an efficient MHD convertor, it is desirable that most of this power be delivered to the load as opposed to heating the electrons. The results show that at near full ionisation, there is still a maximum $\beta$ limit, but this limit becomes progressively larger when the seed density is carefully chosen, thereby allowing operation with larger, still stable, magnetic fields. This, in turn, is shown to decrease the ratio $S_{\Omega }/S_{L}$. In other words, high field becomes a potentially winning strategy, lowering $S_{\Omega }/S_{L}$ from values well above unity to less than 20 %.

The second figure of merit is the electrical conductivity $\sigma$, defined by $\sigma =n_{e}e^{2}/m_{e}\nu _{M}$ with $n_{e}$ the electron seed number density. Intuitively, one wants a large conductivity. Higher conductivity implies a higher quality plasma, allowing larger currents to flow, which in turn should produce more efficient energy conversion. The results show that when the stability criterion is taken into account, lower field produces a higher conductivity.

> Figure 1. Schematic diagram of a Hall MHD generator.

The apparently opposing intuitive conclusions regarding the effects of higher magnetic fields cannot be resolved by the analysis presented here. The reason is that the analysis is a local analysis, which determines a stability criterion that must be satisfied at every point along the MHD channel. However, a true measure of ‘improved’ performance can only be determined by a global analysis which takes into account the generator geometry plus engineering constraints. This would predict what fraction of the input power (actually total enthalpy flux) is converted into load power, with high values obviously representing the most desirable outcomes. Such a global analysis has been completed recently and will be presented in a future paper. For the present, one shall have to be content with a demonstration and explanation of conflicting conclusions on the impact of high magnetic field on MHD convertor performance. Even so, if the stability of two-fluid MHD plasmas with temperature ratios of the order of 10 for various magnetic fields can be verified experimentally, this would represent a major first step for the reconsideration of MHD energy conversion as a topping cycle in nuclear power plants, both fission and fusion.

### 1.6. Details of the analysis

The main contributions of the work have been defined in general terms. To proceed, it now makes sense to present a brief overview of the theoretical details resulting in these contributions. The theoretical strategy is as follows.

(i) For mathematical simplicity, the analysis focuses on a linear Hall MHD generator. The linear Hall configuration is attractive because it can be modified into a cylindrical disk geometry, with huge savings in reducing the number of required electrodes, from many hundreds to two. A simple schematic drawing of a Hall generator is illustrated in figure 1. It is a rectangular channel with a slowly expanding cross-section along its length. There are many pairs of electrodes on each side panel which, for a Hall generator, are all shorted out on top and bottom. The output Hall voltage, driving the load, is generated between the first and last pair of shorted electrodes.

(ii) The main goal of the analysis is to calculate, from first principles, the marginal stability criterion for the Velikhov-ionization instability, valid in the regime of nearly full ionisation of the seed gas.

(iii) We assume the modes have short wavelengths and thus vary rapidly over the characteristic length scale of the generator. This assumption allows the introduction a standard multiple length scale expansion, which basically divides the plasma into a series of corresponding narrow axial slices. For each slice, the equilibrium properties are essentially homogenous in space. The implication is that each slice can be treated separately and independently; that is, the analysis is transformed into a local stability analysis. Suppression of the ionisation instability requires that the stability criterion be satisfied in each and every slice.

(iv) The slice location with the most severe constraint depends, in general, on the shape of the channel. One strategy, discussed later in the analysis, is to design the channel shape so that the marginal stability criterion is satisfied locally along the entire length of the generator. Under this strategy, the ionisation instability actually determines the input boundary conditions on the generator.

(v) All properties of the primary gas remain unchanged during the perturbation. The quantities that vary are the electron density, electron temperature, current density and electric field. It is shown that the mathematical validity of the unperturbed primary gas assumption follows from the short wavelength assumption.

## 2. Physical picture of the ionisation instability

The derivation of the marginal stability criterion for the ionisation instability requires a considerable amount of analysis. It is, therefore, useful to first provide a qualitative physical explanation of the instability, describing the key phenomena that drive and stabilise the mode.

We begin by assuming that a closed cycle MHD generator is operating in a normal quiescent mode, when the seed electrons experience a small positive localised perturbation in temperature. Since the energy equilibration time with the primary gas is assumed to be long, then corresponding collisions will have a negligible effect restoring the electrons to their original temperature as they flow along the generator.

If the seed plasma is weakly ionised, then by definition, the electron density $n_{e}$ is small compared with the original seed density $n_{s}: n_{e}\ll n_{s}$. Thus, in accordance with Saha’s ionisation equation, which determines $n_{e}$, the positive electron temperature perturbation will cause a corresponding increase in electron density. This density increase leads to a decrease in the electrical resistivity. Specifically, if electron–primary gas momentum collisions $\nu _{ep}$ are dominant, then the total momentum exchange collision frequency $\nu _{M}\approx \nu _{ep}$ and the electrical resistivity scales as

**Equation (2.1)**

\begin{equation} \eta =\frac{m_{e}\nu _{M}}{e^{2}n_{e}}\approx \frac{m_{e}\nu _{ep}}{e^{2}n_{e}}\approx \frac{m_{e}n_{p}\overline{\sigma }_{ep}v_{Te}}{e^{2}n_{e}}\propto \frac{T_{e}^{1/2}}{n_{e}}. \end{equation}

Here, $n_{p}$ is the primary gas number density, $\overline{\sigma }_{ep}$ is the electron–primary gas collision cross-section and $v_{Te}=(2kT_{e}/m_{e})^{1/2}$ is the electron thermal velocity. Because of the strong exponential $T_{e}$ behaviour in the Saha equation, the resulting $n_{e}(T_{e})$ density increase dominates and the resistivity decreases.

Now, the decreased resistivity allows more current to flow in the plasma, which in turn increases the Ohmic heating. The increase in Ohmic heating tends to further heat the electrons. This is equivalent to positive feedback and is the source of the ionisation instability.

How does nearly full ionisation help stabilise the mode? If the plasma is nearly fully ionised, then $n_{e}\approx n_{s}$. Thus, Saha’s equation implies that a small positive increase in electron temperature will bring the electrons even closer to full ionisation $n_{e}\rightarrow n_{s}$. However, this cannot produce a significant increase in electron density, since the plasma already starts off almost fully ionised: $n_{e}\approx n_{s}=\text{ constant}$. The net result is that in this situation, the electrical resistivity actually increases under a positive temperature perturbation,

**Equation (2.2)**

\begin{equation} \eta =\frac{m_{e}\nu _{M}}{e^{2}n_{e}}\approx \frac{m_{e}\nu _{ep}}{e^{2}n_{e}}\propto \frac{T_{e}^{1/2}}{n_{e}}\approx \frac{T_{e}^{1/2}}{n_{s}}\propto T_{e}^{1/2}. \end{equation}

The increase in resistivity decreases the plasma current and corresponding Ohmic heating. The plasma becomes cooler which tends to restore the plasma to its original temperature. This negative feedback produces the stabilisation at near full ionisation.

Marginal stability occurs when the enhanced ionisation is exactly balanced by the increased cooling. Because of the exponential temperature dependence in Saha’s equation, the marginal stability transition point is a strong function of temperature that always occurs at a point near full ionisation. In terms of the overall generator design, the theoretical analysis that follows shows that for a given applied magnetic field, stabilisation of the ionisation instability plays a major role in setting the value of the seed density. In more detail, the combination of the marginal stability criterion plus the practical requirement of an industrial relevant load power density $S_{L}=-\boldsymbol{E}\boldsymbol{\cdot} \boldsymbol{J}\sim 100\text{ MW}\,\mathrm{m}^{-3}$ represent two design constraints. Their simultaneous solution leads to specific values for the electron temperature and seed density. The electron density, which is nearly equal to the seed density, is determined by the ionisation production rate, as obtained from the Saha equation. These constraints and solutions are quantified in the following analysis.

## 3. The starting model

The starting model describing both equilibrium and stability of the MHD plasma is a set of coupled three-dimensional (3-D) nonlinear equations for two separate fluids, one the primary gas neutrals (e.g. argon) and the second, the seed gas electrons (e.g. potassium). The goal is to determine the conditions for marginal stability; that is, the mode deteriorates performance to such a large degree that it is more important to learn how to avoid it in the first place, as opposed to calculating an accurate growth rate (which can be easily estimated) or following the catastrophic nonlinear dynamics of the unstable evolution. Equilibrium operation and marginal stability both require setting $\partial /\partial t=0$. The starting equations for a monatomic primary gas are then given by the following.

### 3.1. Primary MHD fluid

**Equation (3.1)**

\begin{align} &\boldsymbol{\nabla} \boldsymbol{\cdot} \left(n_{p}\it{v}_{p}\right)=0&&\quad \text{Mass},\nonumber\\[6pt] &m_{p}n_{p}\it{v}_{p}\boldsymbol{\cdot} \boldsymbol{\nabla} \it{v}_{p}=\boldsymbol{J}\times \boldsymbol{B}_{0}-\boldsymbol{\nabla} p_{p}&&\quad \text{Momentum},\nonumber\\[6pt] &\frac{3}{2}\boldsymbol{\nabla} \boldsymbol{\cdot} \left(p_{p}\it{v}_{p}\right)+p_{p}\boldsymbol{\nabla} \boldsymbol{\cdot} \it{v}_{p}=\frac{3}{2}\nu _{E}n_{e}\left(kT_{e}-kT_{p}\right)&&\quad \text{Energy}. \end{align}

### 3.2. Electron fluid

**Equation (3.2)**

\begin{align} &\frac{n_{e}^{2}}{n_{s}-n_{e}}=\left(\frac{2\pi m_{e}kT_{e}}{h^{2}}\right)^{3/2}\exp \left(-E_{I}/kT_{e}\right)&&\quad \text{Mass}-\text{Saha equation},\nonumber\\[6pt] &\boldsymbol{E}+\it{v}_{p}\times \boldsymbol{B}_{0}=\eta \boldsymbol{J}+\frac{\boldsymbol{J}\times \boldsymbol{B}_{0}}{en_{e}}&&\quad \text{Momentum}-\mathrm{Ohm}'\text{s law},\nonumber\\[6pt] &\frac{3}{2}\nu _{E}n_{e}\left(kT_{e}-kT_{p}\right)=\eta J^{2}&&\quad \text{Energy}-\text{Electron energy balance}.\end{align}

### 3.3. Maxwell

**Equation (3.3)**

\begin{align} &\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{J}=0&&\quad \mathrm{Ampere}\text{'s law},\nonumber\\ &\boldsymbol{\nabla} \times \boldsymbol{E}=0&&\quad \mathrm{Faraday}\text{'s law},\nonumber\\ &n_{i}=n_{e}&&\quad \mathrm{Poisson}\text{'s equation}-\text{charge neutrality}. \end{align}

Note that the primary and electron fluid variables are denoted by the subscript ‘*p*’ and ‘*e*’, respectively. The quantity $E_{I}$ is the ionisation potential of the seed gas, $n_{s}$ is the density of the initial unionised seed gas and $\boldsymbol{B}_{0}$ is the applied vacuum magnetic field. The resistivity $\eta$ and temperature equilibration frequency $\nu _{E}$ for a monatomic gas are related to the electron momentum exchange frequency $\nu _{M}$ in the usual way,

**Equation (3.4)**

\begin{align} \nu _{M}&=\nu _{ep}+\nu _{en}+\nu _{ei}\approx \nu _{ep},\nonumber\\[5pt] \eta &=\frac{m_{e}\nu _{M}}{n_{e}e^{2}},\nonumber\\[5pt] \nu _{E}&=2\frac{m_{e}}{m_{p}}\nu _{M}. \end{align}

Here, the separate contributions to $\nu _{M}$ represent electron collisions with (a) the primary gas ($\nu _{ep}$), (b) the actual unionised seed particles ($\nu _{en}$) and (c) the seed ions ($\nu _{ei}$). Some discussion of the physics is warranted.

(i) The use of fluid equations assumes that all collision frequencies are large compared with the characteristic MHD time scale: $\nu _{E}\gg v_{p}/L$ with $L$ the macroscopic generator length scale. This condition is well satisfied in all regimes of interest.

(ii) With respect to momentum collisions, the electron–primary gas interactions dominate in the operational regime of interest: $\nu _{ep}\gg \nu _{en},\nu _{ei}$. For mathematical simplicity, one thus assumes $\nu _{M}\approx \nu _{ep}$ in the following analysis. Even so, note that the analysis is generalised to include the full $\nu _{M}$, and some results are included for comparison. Also, an expression for $\nu _{ep}$ is given shortly.

(iii) The additional unknown $n_{s}$ appears in the Saha equation, which as stated represents the initial, total unionised number density of the seed gas. The actual number density of unionised seed particles is $n_{n}=n_{s}-n_{i}=n_{s}-n_{e}$. Now, because of the high collisionality, it follows that all species, except the very small mass electrons, have essentially the same velocity and temperature

**Equation (3.5)**

\begin{equation} v_{i}\approx v_{n}\approx v_{p},\quad T_{i}\approx T_{n}\approx T_{p}.\end{equation}

The neutral and ionised seed particles are dragged along and thermally equilibrated with the much denser primary gas. After a short calculation using the mass conservation equations (including ionisation and recombination) for the heavy species, it can be easily shown that the quantity $n_{s}$ satisfies $\boldsymbol{\nabla} \boldsymbol{\cdot} (n_{s}\it{v}_{p})=0$. Therefore, assuming that all species have the same uniform cross-sectional profile at the generator inlet, it follows that

**Equation (3.6)**

\begin{equation} \frac{n_{s}}{n_{s0}}=\frac{n_{p}}{n_{p0}},\end{equation}

with a ‘0’ subscript denoting inlet value. This is the simple, required information relating $n_{s}$ to the basic unknown $n_{p}$.

(i) The next point of interest is the assumption that only the vacuum magnetic field enters the analysis. This too is a very good approximation since simple scaling relations show that the induced magnetic field $B_{{\rm ind}}$ due to the generator currents is much smaller than the vacuum field: $B_{{\rm ind}}\ll B_{0}$. One consequence of this approximation is that the only non-trivial information contained in Ampere’s law is the relation $\boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{J}=0$.

(ii) Another point to observe is that viscosity and thermal conduction are neglected in the model. This too is a good approximation since these effects are small in the core of the plasma where all profiles are essentially uniform across the cross-section. They are, however, important near the MHD channel walls where narrow boundary layers form. Since these represent energy loss mechanisms, they should be maintained when assessing the overall performance of an MHD generator, but they do not play an important role in the development of the ionisation instability in the core plasma.

(iii) A further key point of physics is that the electron energy equation describes a basic balance between preferential electron ohmic heating and the resulting temperature difference between electrons and the primary gas. Convection and compression effects can be shown to be small, and are therefore neglected.

(iv) Lastly, note that various similar, although not identical, forms of the starting equations have appeared in the literature (see for instance Mitchner & Kruger 1973; Rosa 1987; Messerle 1994) – there is nothing dramatically different from the present starting model. Still, it can be shown by a rigorous mathematical maximal ordering procedure that the model described above is entirely self-consistent and contains all the information required to calculate marginal stability of the ionisation instability.

The analysis will show that stability ultimately requires a relatively low fraction of seed gas to primary gas. Therefore, it should be a good approximation to assume that the momentum and energy exchange collision frequencies are dominated by electron–primary gas interactions. The validity of this approximation is tested at the end of the calculation. However, it is not explicitly employed during the analysis.

A final point is that while the primary fluid mass, momentum and energy relations have been listed as part of the starting equations, they actually do not enter the stability analysis. The reason, as stated, is that these relations are needed to determine the slow axial dependence of the equilibrium quantities, but are not directly required for the short wavelength stability analysis. The relations yield expressions for the perturbed fluid density, velocity and primary temperature, which when compared with other terms in the electron equations can be shown to be small by $1/k_{0}L\ll 1$, where $k_{0}$ is the wavenumber of the unstable mode. These quantities can be calculated, if desired, after the analysis of the electron equations. Physically, the ionisation instability is predominantly an electron phenomenon involving ionisation and electron energy balance. The primary fluid does not play a significant role.

### 3.4. Equilibrium

The first step in the analysis is to examine the equilibrium properties of the plasma. In addition to expressing all quantities in terms of the input velocity and magnetic field, it will also be of use in the stability analysis to derive a relationship between the temperature difference, Hall parameter and ionisation fraction. As stated, the analysis focuses primarily on the behaviour of the electrons because of the $k_{0}L\gg 1$ assumption.

### 3.5. Electron Ohm’s law equilibrium properties

We begin by writing Ohm’s law in component form for a linear MHD Hall generator. The flow is along the $x$ axis and the magnetic field is uniform and in the $z$ direction. For a Hall generator, the electrodes are shorted out in the $y$ direction implying that $E_{y}=0$. Also, no voltage or current flow occurs along the magnetic field direction, so that $E_{z}=J_{z}=0$. Ohm’s law reduces to

**Equation (3.7)**

\begin{align} E_{x}&=\eta J_{x}+\frac{J_{y}B_{0}}{en_{e}},\nonumber\\[6pt] -v_{p}B&=\eta J_{y}-\frac{J_{x}B_{0}}{en_{e}},\nonumber\\[6pt] E_{z}&=0. \end{align}

As assumed, the fluid velocity, $\it{v}_{p}\approx v_{p}\boldsymbol{e}_{x}$, of the primary gas will remain unchanged during the development of the instability.

Next, one introduces definitions of several basic parameters entering the analysis,

**Equation (3.8)**

\begin{align} \nu _{M}&=\nu _{ep}=n_{p}\overline{\sigma }_{ep}v_{Te}=n_{p}\overline{\sigma }_{ep}\left(\frac{2kT_{e}}{m_{e}}\right)^{1/2}&&\quad \text{Collision frequency},\quad \nonumber\\[6pt] \eta &=\frac{m_{e}\nu _{M}}{e^{2}n_{e}}&&\quad \text{Resistivity},\nonumber\\[6pt] \beta &=\frac{B_{0}}{en_{e}\eta }=\frac{\Omega _{e}}{\nu _{M}}&&\quad \text{Hall parameter},\nonumber\\[6pt] M&=\left(\frac{3}{5}\frac{m_{p}v_{p}^{2}}{kT_{p}}\right)^{1/2}&&\quad \text{Mach number}.\end{align}

Here, $\overline{\sigma }_{ep}$ is the known cross-section for momentum exchange collisions between the electrons and the primary gas. Also, the Mach number is important since the flow must be supersonic, $M> 1$, for good energy conversion in a Hall MHD generator.

In addition, define an equilibrium load resistivity $\eta _{L}$ as

**Equation (3.9)**

\begin{equation} \eta _{L}\left(x\right)=-\frac{E_{x}}{J_{x}}, \end{equation}

which is then normalised as

**Equation (3.10)**

\begin{equation} Z\left(x\right)=\frac{\eta _{L}}{\eta }. \end{equation}

The quantity $Z(x)$ replaces $E_{x}(x)$ as one of the basic unknowns and is introduced to simplify the analysis.

Using these relations, one obtains, after a straightforward calculation, the desired expressions for the current densities and electric fields in terms of the fluid velocity,

**Equation (3.11)**

\begin{align} J_{x}&=\left[\frac{\beta ^{2}}{\beta ^{2}+1+Z}\right]en_{e}v_{p},\nonumber\\[6pt] J_{y}&=-\frac{1+Z}{\beta }J_{x}=-\left[\frac{\beta \left(1+Z\right)}{\beta ^{2}+1+Z}\right]en_{e}v_{p},\nonumber\\[6pt] \frac{E_{x}}{\eta }&=-ZJ_{x}=-\left[\frac{\beta ^{2}Z}{\beta ^{2}+1+Z}\right]en_{e}v_{p},\nonumber\\[6pt] \frac{E_{y}}{\eta }&=0. \end{align}

Also of interest are expressions for the power density to the load $S_{L}$, the Ohmic power density $S_{\Omega }$ and the total input power density converted to electricity $S_{C}=S_{L}+S_{\Omega }$:

**Equation (3.12)**

\begin{align} S_{L}&=-\boldsymbol{E}\boldsymbol{\cdot} \boldsymbol{J}=-E_{x}J_{x}=m_{e}n_{e}\nu _{M}v_{p}^{2}\frac{\beta ^{4}Z}{\left(\beta ^{2}+1+Z\right)^{2}},\nonumber\\[4pt] S_{\Omega }&=\eta J^{2}=\eta \left(J_{x}^{2}+J_{y}^{2}\right)=m_{e}n_{e}\nu _{M}v_{p}^{2}\frac{\beta ^{2}\left[\beta ^{2}+\left(1+Z\right)^{2}\right]}{\left(\beta ^{2}+1+Z\right)^{2}},\nonumber\\[4pt] S_{C}&=S_{L}+S_{\Omega }=m_{e}n_{e}\nu _{M}v_{p}^{2}\frac{\beta ^{2}\left(1+Z\right)}{\beta ^{2}+1+Z}. \end{align}

### 3.6. Electron energy balance

We now focus on calculating the temperature difference due to Ohmic heating. The desired relation is obtained by substituting into the energy balance relation given by (3.2) and repeated here for convenience:

**Equation (3.13)**

\begin{equation} \frac{3}{2}\nu _{E}n_{e}\left(kT_{e}-kT_{p}\right)=\eta J^{2}. \end{equation}

Substituting into this relation, one can evaluate the temperature ratio of interest $T_{e}/T_{p}$. The result is

**Equation (3.14)**

\begin{equation} \frac{T_{e}}{T_{p}}-1=\frac{2\eta J^{2}}{3\nu _{E}n_{e}kT_{p}}=\left(\frac{5M^{2}}{9}\right)\frac{\beta ^{2}\left[\beta ^{2}+\left(1+Z\right)^{2}\right]}{\left(\beta ^{2}+1+Z\right)^{2}}. \end{equation}

Note that a large temperature ratio requires high values for the Mach number and Hall parameter.

This information, coupled with the Saha equation, defines all the equilibrium information required for the stability analysis.

## 4. Stability – the marginal stability boundary

The stability analysis is carried out by linearising all quantities about their equilibrium values using a standard multiple length scale expansion. Assume a slow length scale $\boldsymbol{r}_{s}$ to describe variations along the long equilibrium length scale. Similarly, introduce a fast length scale $\boldsymbol{r}_{f}$ to describe the short wavelength variations associated with the instability. The implication is that any quantity $Q_{Tot}(\boldsymbol{r},t)$ can be linearised as follows:

**Equation (4.1)**

\begin{align} Q_{Tot}(\boldsymbol{r},t) & =Q\!\left(\boldsymbol{r}\right)+\tilde{Q}(\boldsymbol{r},t)\nonumber\\ & =Q(\boldsymbol{r}_{s})+\tilde{Q}(\boldsymbol{r}_{f},\boldsymbol{r}_{s},t)\nonumber\\ & =Q(\boldsymbol{r}_{s})+\tilde{Q}(\boldsymbol{r}_{s})\exp\!\left(\gamma t+ik_{x}x_{f}+ik_{y}y_{f}+ik_{z}z_{f}\right). \end{align}

Here, quantities with a tilde are the linearised perturbations and $\gamma \rightarrow 0$ corresponds to marginal stability.

A key mathematical point is to distinguish the different length scales between equilibrium and stability. This corresponds to the following ordering requirement: $\boldsymbol{\nabla} Q\sim \partial Q/\partial \boldsymbol{r}_{s}\sim Q/L$ and $\boldsymbol{\nabla} \tilde{Q}\sim \partial \tilde{Q}/\partial \boldsymbol{r}_{s}+\partial \tilde{Q}/\partial \boldsymbol{r}_{f}\sim \tilde{Q}/L+ik_{0}\tilde{Q}\approx ik_{0}\tilde{Q}$ with $k_{0}^{2}=k_{x}^{2}+k_{y}^{2}+k_{z}^{2}$. Because of the short wavelength assumption, each equilibrium quantity $Q(\boldsymbol{r}_{s})$ can be viewed as a ‘constant’ on the short $\boldsymbol{r}_{f}$ stability length scale. The derivation of the stability boundary thus becomes purely algebraic (i.e. no differential equations to solve). It has the form of a local criterion that must be satisfied separately at each point $\boldsymbol{r}_{s}$ along the MHD channel. Violation at any location will lead to the development of a short wavelength ionisation instability.

The strategy of the analysis is to express all perturbed quantities in terms of the perturbed electron density. Once accomplished, it is then straightforward to determine the marginal stability dispersion relation. A useful order to carry out the analysis is described as follows.

### 4.1. The Saha equation

The Saha equation, given by (3.2), determines the relation between $\tilde{T}_{e}$ and $\tilde{n}_{e}$. Linearisation of the Saha equation yields

**Equation (4.2)**

\begin{equation} \frac{\tilde{T}_{e}}{T_{e}}=2\alpha \frac{\tilde{n}_{e}}{n_{e}}, \end{equation}

where

**Equation (4.3)**

\begin{align} \alpha &=\frac{1}{2}\frac{N}{T_{e}\left(\mathrm{d}N/\mathrm{d}T_{e}\right)}=\frac{1}{2}\left(\frac{2kT_{e}}{3kT_{e}+2E_{I}}\right)\left(\frac{2-f_{I}}{1-f_{I}}\right)\nonumber\\[4pt]&\approx \left(\frac{kT_{e}}{2E_{I}}\right)\left(\frac{2-f_{I}}{1-f_{I}}\right)\propto \frac{1}{1-f_{I}} \end{align}

and $f_{I}=n_{e}/n_{s}$ is the equilibrium fraction of seed ionisation. The approximate expression makes use of the fact that $kT_{e}/E_{I}\ll 1$. The value of $\alpha$ is critical in that it is a direct measure of the degree of ionisation of the seed gas. Small $\alpha \ll 1$ implies small to modest ionisation fraction: $f_{I}< 1$. Moderate to large $\alpha > 1$ implies nearly full ionisation: $f_{I}\rightarrow 1$. Also, as stated, the perturbed $\tilde{n}_{s}$ should be, and is, neglected.

### 4.2. The perturbed collision frequency

In the analysis, it is necessary to evaluate the perturbed collision frequency plus several related quantities. Consider first the momentum exchange collision frequency defined by $\nu _{M}=\nu _{ep}$. A straightforward calculation leads to

**Equation (4.4)**

\begin{equation} \tilde{\nu }_{M}=\tilde{\nu }_{ep}=n_{p}\overline{\sigma }_{p}v_{Te}\left(\frac{1}{2}\frac{\tilde{T}_{e}}{T_{e}}\right)=\nu _{ep}\left(\frac{1}{2}\frac{\tilde{T}_{e}}{T_{e}}\right). \end{equation}

Next, $\tilde{T}_{e}/T_{e}$ from (4.2) is substituted, yielding the required relation between $\tilde{\nu }_{M}$ and $\tilde{n}_{e}$,

**Equation (4.5)**

\begin{equation} \frac{\tilde{\nu }_{M}}{\nu _{M}}=\alpha \frac{\tilde{n}_{e}}{n_{e}}. \end{equation}

From (4.5), it then follows that the perturbed Hall parameter is given by

**Equation (4.6)**

\begin{equation} \frac{\tilde{\beta }}{\beta }=\frac{\left(\Omega _{e}/\nu _{M}\right)\left(-\tilde{\nu }_{M}/\nu _{M}\right)}{\left(\Omega _{e}/\nu _{M}\right)}=-\frac{\tilde{\nu }_{M}}{\nu _{M}}=-\alpha \frac{\tilde{n}_{e}}{n_{e}}. \end{equation}

Similarly, one can easily derive the perturbed resistivity. Since the resistivity is defined as $\eta =m_{e}\nu _{M}/n_{e}e^{2}$, this leads to

**Equation (4.7)**

\begin{equation} \frac{\tilde{\eta }}{\eta }=\frac{\tilde{\nu }_{M}}{\nu _{M}}-\frac{\tilde{n}_{e}}{n_{e}}=\left(\alpha -1\right)\frac{\tilde{n}_{e}}{n_{e}}. \end{equation}

Lastly, the perturbed energy equilibration time can be written as

**Equation (4.8)**

\begin{equation} \frac{\tilde{\nu }_{E}}{\nu _{E}}=\frac{\tilde{\nu }_{M}}{\nu _{M}}=\alpha \frac{\tilde{n}_{e}}{n_{e}}. \end{equation}

### 4.3. Maxwell’s equations

Maxwell’s equations lead to relationships between the components of the perturbed electric field and current density in terms of the wavenumbers. These relations are given by

**Equation (4.9)**

\begin{align} \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{J} & =0\rightarrow k_{x}\tilde{J}_{x}+k_{y}\tilde{J}_{y}+k_{z}\tilde{J}_{z}=0,\nonumber\\ \boldsymbol{\nabla} \times \boldsymbol{E} & =0\rightarrow k_{x}\tilde{E}_{y}-k_{y}\tilde{E}_{x}=0\quad \textrm{and}\quad k_{x}\tilde{E}_{z}-k_{z}\tilde{E}_{x}=0.\end{align}

Note that the term $\partial \boldsymbol{B}/\partial t$ has been neglected in Faraday’s law based on the well-satisfied assumption that the induced currents in both equilibrium and stability are small, plus the focus on marginal stability. The magnetic field is a pure static vacuum field.

### 4.4. The perturbed currents

The perturbed currents are found from the MHD Ohm’s law, neglecting variations in the primary gas velocity: $\tilde{v}_{p}=0$. After some straightforward algebra using (4.9) plus the $z$ component of Ohm’s law, one finds that $\tilde{J}_{y},\tilde{J}_{z},\tilde{E}_{y},\tilde{E}_{z}$ can be expressed in terms of $\tilde{J}_{x},\tilde{E}_{x}$ as follows:

**Equation (4.10)**

\begin{align} \tilde{E}_{y}&=\frac{k_{y}}{k_{x}}\tilde{E}_{x},\nonumber\\ \tilde{E}_{z}&=\frac{k_{z}}{k_{x}}\tilde{E}_{x},\nonumber\\ \tilde{J}_{z}&=\frac{k_{z}}{k_{x}}\frac{\tilde{E}_{x}}{\eta },\nonumber\\ \tilde{J}_{y}&=-\frac{k_{x}}{k_{y}}\tilde{J}_{x}-\frac{k_{z}^{2}}{k_{x}k_{y}}\frac{\tilde{E}_{x}}{\eta }. \end{align}

Using these relations, one can write the $x$ and $y$ components of Ohm’s law in terms of $\tilde{J}_{x},\,\tilde{E}_{x}$ as

**Equation (4.11)**

\begin{align} \left(1-\beta \frac{k_{x}}{k_{y}}\right)\tilde{J}_{x}&-\left(1+\frac{\beta k_{z}^{2}}{k_{x}k_{y}}\right)\frac{\tilde{E}_{x}}{\eta }=\left[-\left(\alpha -1\right)J_{x}+\beta J_{y}\right]\frac{\tilde{n}_{e}}{n_{e}},\nonumber\\[3pt]\left(\frac{k_{x}}{k_{y}}+\beta \right)\tilde{J}_{x}&+\left(\frac{k_{y}}{k_{x}}+\frac{k_{z}^{2}}{k_{x}k_{y}}\right)\frac{\tilde{E}_{x}}{\eta }=\left[\left(\alpha -1\right)J_{y}+\beta J_{x}\right]\frac{\tilde{n}_{e}}{n_{e}}.\end{align}

Equation (4.11) can be solved for $\tilde{J}_{x}$ and $\tilde{E}_{x}$. These results are then back substituted to obtain expressions for several other quantities of interest appearing in the stability analysis. After a slightly tedious calculation, one obtains the required results,

**Equation (4.12)**

\begin{align} \tilde{J}_{x}&=\frac{1}{1+\beta ^{2}\left(k_{z}^{2}/k_{0}^{2}\right)}\left\{\left(\alpha -1\right)\frac{k_{y}}{k_{0}}J_{\bot }+\beta \frac{k_{y}}{k_{0}}J_{\parallel }\vphantom{\frac{k_{z}^{2}}{k_{0}^{2}}}\right.\nonumber\\& \qquad\qquad\qquad\qquad\left. +\frac{k_{z}^{2}}{k_{0}^{2}}\left[\left(\beta ^{2}-\alpha +1\right)J_{x} +\alpha \beta J_{y}\right]\right\}\frac{\tilde{n}_{e}}{n_{e}},\nonumber\\[4pt]\frac{\tilde{E}_{x}}{\eta }&=-\frac{1}{1+\beta ^{2}\left(k_{z}^{2}/k_{0}^{2}\right)}\frac{k_{x}}{k_{0}}\left[\left(\beta ^{2}-\alpha +1\right)J_{\parallel }+\alpha \beta J_{\bot }\right]\frac{\tilde{n}_{e}}{n_{e}},\nonumber\\[4pt]\tilde{J}_{y}&=\frac{1}{1+\beta ^{2}\left(k_{z}^{2}/k_{0}^{2}\right)}\left\{-\left(\alpha -1\right)\frac{k_{x}}{k_{0}}J_{\bot }-\beta \frac{k_{x}}{k_{0}}J_{\parallel }\vphantom{\frac{k_{z}^{2}}{k_{0}^{2}}}\right.\nonumber\\& \qquad\qquad\qquad\qquad\left. +\frac{k_{z}^{2}}{k_{0}^{2}}\left[\left(\beta ^{2}-\alpha +1\right)J_{y}-\alpha \beta J_{x}\right]\right\}\frac{\tilde{n}_{e}}{n_{e}},\nonumber\\[4pt]J_{x}\tilde{J}_{x}+J_{y}\tilde{J}_{y}&=\frac{1}{1+\beta ^{2}\left(k_{z}^{2}/k_{0}^{2}\right)}\left[-\left(\alpha -1\right)J_{\bot }^{2}-\beta J_{\bot }J_{\parallel }+\frac{k_{z}^{2}}{k_{0}^{2}}\left(\beta ^{2}-\alpha +1\right)J^{2}\right]\frac{\tilde{n}_{e}}{n_{e}}. \end{align}

Here,

**Equation (4.13)**

\begin{align} k_{0}^{2}&=k_{x}^{2}+k_{y}^{2}+k_{z}^{2},\nonumber\\[2pt] J^{2}&=J_{x}^{2}+J_{y}^{2}=\left(J_{\bot }^{2}+J_{\parallel }^{2}\right)/\left(1-k_{z}^{2}/k_{0}^{2}\right),\nonumber\\[2pt] J_{z}&=0,\nonumber\\[2pt] J_{\bot }&=\frac{1}{k_{0}}\left(\boldsymbol{e}_{z}\boldsymbol{\cdot} \boldsymbol{k}\times \boldsymbol{J}\right)=\frac{1}{k_{0}}\left(k_{x}J_{y}-k_{y}J_{x}\right),\nonumber\\[2pt] J_{\parallel }&=\frac{1}{k_{0}}\left(\boldsymbol{k}\boldsymbol{\cdot} \boldsymbol{J}\right)=\frac{1}{k_{0}}\left(k_{x}J_{x}+k_{y}J_{y}\right). \end{align}

### 4.5. The energy balance equation

The last equation of interest is the energy balance equation. For convenience, this equation is repeated here, adding the standard time variation term on the left-hand side,

**Equation (4.14)**

\begin{equation} E_{I}\frac{\partial n_{e}}{\partial t}=\eta J^{2}-\frac{3}{2}\nu _{E}n_{e}\left(kT_{e}-kT_{p}\right).\end{equation}

The time variation term vanishes in the limit of marginal stability, but is included to determine the correct sign for the stability boundary.

Linearising this equation leads to

**Equation (4.15)**

\begin{equation} \gamma E_{I}\tilde{n}_{e}=J^{2}\tilde{\eta }+2\eta \big(J_{x}\tilde{J}_{x}+J_{y}\tilde{J}_{y}\big)-\frac{3}{2}\left(kT_{e}-kT_{p}\right)\left(\tilde{\nu }_{E}n_{e}+\nu _{E}\tilde{n}_{e}\right)-\frac{3}{2}\nu _{E}n_{e}k\tilde{T}_{e}.\end{equation}

As before, neglect the variation of the primary gas temperature $\tilde{T}_{p}\ll \tilde{T}_{e}$. Each of the terms in (4.15) can now be expressed in terms of $\tilde{n}_{e}$. A short calculation yields

**Equation (4.16)**

\begin{align} &\left(\gamma n_{e}E_{I}\right)\frac{\tilde{n}_{e}}{n_{e}} =\left\{\eta J^{2}\left(\alpha -1+2K\right)-\frac{3}{2}n_{e}\nu _{E}\left[\left(3\alpha +1\right)\left(kT_{e}-kT_{p}\right)+2\alpha kT_{P}\right]\right\}\frac{\tilde{n}_{e}}{n_{e}}, \end{align}

with

**Equation (4.17)**

\begin{equation} K\left(k_{x},k_{y},k_{z}\right)=\frac{1}{1+\beta ^{2}\left(k_{z}^{2}/k_{0}^{2}\right)}\left[-\left(\alpha -1\right)\frac{J_{\bot }^{2}}{J^{2}}-\beta \frac{J_{\bot }J_{\parallel }}{J^{2}}+\frac{k_{z}^{2}}{k_{0}^{2}}\left(\beta ^{2}-\alpha +1\right)\right]. \end{equation}

Taking the limit $\gamma \rightarrow 0$, one sees that the dispersion relation defining the condition for marginal stability is given by

**Equation (4.18)**

\begin{equation} \eta J^{2}\left(\alpha -1+2K\right)-\frac{3}{2}n_{e}\nu _{E}\left[\left(3\alpha +1\right)\left(kT_{e}-kT_{p}\right)+2\alpha kT_{P}\right]\leq 0\quad \text{for stability}.\end{equation}

This is the required relation with simplifications to follow.

Before proceeding with simplification, this subsection is completed by providing the justification for neglecting the perturbed quantities associated with the primary fluid. An important point to recognise is that the maximal ordering assumed in the starting equations requires that $\beta \sim \alpha \sim M^{2}\sim O(1)$. With this assumption, it follows that all the electron perturbations just calculated are of the same order,

**Equation (4.19)**

\begin{equation} \frac{\tilde{n}_{e}}{n_{e}}\sim \frac{\tilde{T}_{e}}{T_{e}}\sim \frac{\tilde{J}_{x}}{J_{x}}\sim \frac{\tilde{J}_{y}}{J_{y}}.\end{equation}

If one now uses this ordering in the primary fluid mass, momentum and energy equations, it follows that

**Equation (4.20)**

\begin{equation} \frac{\tilde{n}_{p}}{n_{p}}\sim \frac{\tilde{v}_{p}}{v_{p}}\sim \frac{\tilde{T}_{p}}{T_{p}}\sim \frac{1}{k_{0}L}\frac{\tilde{n}_{e}}{n_{e}}. \end{equation}

We see that all primary perturbations are smaller by $1/k_{0}L$ than the electron perturbations. This is the justification for neglecting them in the derivation of the dispersion relation.

## 5. Stability – simplifying the dispersion relation

The dispersion relation can be substantially simplified. There are two steps. First, all quantities should be expressed in terms of $M^{2},\,\beta,\,\alpha$. This is straightforward. Second, one has to determine and substitute the most unstable wavenumber. This requires a little work.

We begin with determining the most unstable wavenumber. Observe that the individual wavenumber components only appear in the coefficient $K=K(k_{x},k_{y},k_{z})$. The sign of the inequality in (4.18) is such that the most unstable mode corresponds to choosing the components of the wavenumber to make $K=K_{\max }$. This requires some tedious algebra and the following procedure is a useful strategy.

The components of the wavenumber can be written as

**Equation (5.1)**

\begin{align} &k_{z}=k_{0}\sin \phi, \nonumber\\ &k_{x}=k_{0}\sin \zeta \cos \phi, \nonumber\\ &k_{y}=k_{0}\cos \zeta \cos \phi, \nonumber\\ &k_{x}^{2}+k_{y}^{2}+k_{z}^{2}=k_{0}^{2}. \end{align}

These expressions are substituted into $K$, leading to

**Equation (5.2)**

\begin{equation} K=\frac{1}{1+\beta ^{2}\sin ^{2}\phi }\left\{\left[-\left(\alpha -1\right)\frac{J_{\bot }^{2}}{J^{2}}-\beta \frac{J_{\bot }J_{\parallel }}{J^{2}}\right]+\left(\beta ^{2}-\alpha +1\right)\sin ^{2}\phi \right\}, \end{equation}

with

**Equation (5.3)**

\begin{align} &J_{\parallel }=\frac{\boldsymbol{k}\boldsymbol{\cdot} \boldsymbol{J}}{k_{0}}=\frac{k_{x}J_{x}+k_{y}J_{y}}{k_{0}}=\left(J_{x}\sin \zeta +J_{y}\cos \zeta \right)\cos \phi, \nonumber\\&J_{\bot }=\frac{\boldsymbol{e}_{z}\boldsymbol{\cdot} \boldsymbol{k}\times \boldsymbol{J}}{k_{0}}=\frac{k_{x}J_{y}-k_{y}J_{x}}{k_{0}}=\left(-J_{x}\cos \zeta +J_{y}\sin \zeta \right)\cos \phi, \nonumber\\&J_{\parallel }^{2}+J_{\bot }^{2}=\left(J_{x}^{2}+J_{y}^{2}\right)\cos ^{2}\phi =J^{2}\cos ^{2}\phi. \end{align}

The last equation allows the introduction of a convenient new angle $\chi =\chi (\zeta)$,

**Equation (5.4)**

\begin{align} J_{\parallel }&=-\mathit{J}\cos \chi \cos \phi, \nonumber\\ J_{\bot }&=\mathit{J}\sin \chi \cos \phi. \end{align}

The negative sign in $J_{\parallel }$ implies that a positive $\chi$ maximises rather than minimises $K$. The relation between $\chi$ and $\zeta$ is given by

**Equation (5.5)**

\begin{equation} \tan \chi =\frac{J_{x}-J_{y}\tan \zeta }{J_{x}\tan \zeta +J_{y}}. \end{equation}

Substituting (5.4) into (5.2) leads to

**Equation (5.6)**

\begin{equation} K=\frac{1}{1+\beta ^{2}\sin ^{2}\phi }\left\{\left[-\left(\alpha -1\right)\sin ^{2}\chi +\mathit{\beta }\sin \chi \cos \chi \right]\cos ^{2}\phi +\left(\beta ^{2}-\alpha +1\right)\sin ^{2}\phi \right\}. \end{equation}

Observe that the short wavelength, local approximation, eliminates the explicit appearance of $k_{0}$ in this expression. Thus, the most unstable $K$ is found by maximising with respect to $\chi$ and $\phi$; that is, setting $\partial K/\partial \chi =0$ and $\partial K/\partial \phi =0$. A simple calculation that sets $\partial K/\partial \chi =0$ yields

**Equation (5.7)**

\begin{equation} \tan 2\chi =\frac{\beta }{\alpha -1}.\end{equation}

The value of $K$ reduces to

**Equation (5.8)**

\begin{equation} K=\frac{\left\{\left[\left(\alpha -1\right)^{2}+\beta ^{2}\right]^{1/2}-\left(\alpha -1\right)\right\}\dfrac{1-\sin ^{2}\phi }{2}+\left(\beta ^{2}-\alpha +1\right)\sin ^{2}\phi }{1+\beta ^{2}\sin ^{2}\phi }. \end{equation}

Consider now maximisation with respect to $\phi$. The expression for $K$ is monotonic with respect to $\phi$ with extrema at $\phi =0$ and $\phi =\pi /2$. The most unstable choice is determined by examining the ratio of $K$ at the two extrema. This can be accomplished by rewriting (5.8) as

**Equation (5.9)**

\begin{align} K&=K_{0}\frac{1+\hat{\beta }^{2}\sin ^{2}\phi }{1+\beta ^{2}\sin ^{2}\phi },\nonumber\\ \hat{\beta }^{2}&=\frac{\beta ^{2}-\alpha +1}{K_{0}}-1,\nonumber\\ K_{0}&=\frac{1}{2}\left\{\left[\left(\alpha -1\right)^{2}+\beta ^{2}\right]^{1/2}-\left(\alpha -1\right)\right\}\geq 0.\end{align}

The condition for the maximum to occur at $\phi =0$ requires that $\hat{\beta }^{2}\leq \beta ^{2}$. Conversely, for the maximum to occur at $\phi =\pi /2$ requires that $\hat{\beta }^{2}\geq \beta ^{2}$. A short calculation shows that the $\hat{\beta }^{2}\leq \beta ^{2}$ condition can be written as

**Equation (5.10)**

\begin{equation} \hat{\beta }^{2}\leq \beta ^{2}\rightarrow -\left[2\left(\alpha -1\right)-\left(\beta ^{2}-1\right)\right]^{2}\leq 0.\end{equation}

Clearly, this condition is always satisfied, showing that $\phi =0$ is the most unstable value. This corresponds to $k_{z}=0$, which agrees with intuition. Thus, the most unstable wavenumber leads to a value for $K=K_{\max }$ given by

**Equation (5.11)**

\begin{equation} K_{\max }=K_{0}=\frac{1}{2}\left\{\left[\left(\alpha -1\right)^{2}+\beta ^{2}\right]^{1/2}-\left(\alpha -1\right)\right\}.\end{equation}

The stability condition is now obtained by substituting $K_{\max }$ from (5.11) into the general stability relation given by (4.18). This leads to

**Equation (5.12)**

\begin{equation} \eta J^{2}\left[\left(\alpha -1\right)^{2}+\beta ^{2}\right]^{1/2}-\frac{3}{2}n_{e}\nu _{E}\left[\left(3\alpha +1\right)\left(kT_{e}-kT_{p}\right)+2\alpha kT_{P}\right]\leq 0. \end{equation}

The final simplification is to substitute $\eta J^{2}$ from (3.14). After some straightforward algebra, one finds that the stability condition sets a maximum allowable value for the Hall parameter,

**Equation (5.13)**

\begin{align} \beta ^{2}&\leq 4\alpha \left(2+\frac{1}{\Delta T}\right)\left[1+\alpha \left(1+\frac{1}{\Delta T}\right)\right],\nonumber\\[5pt] \Delta T&=\frac{T_{e}}{T_{p}}-1,\nonumber\\[5pt] \alpha &=\frac{kT_{e}}{2E_{I}}\frac{2-f_{I}}{1-f_{I}}. \end{align}

This is the desired relation. It is convenient in that it is purely algebraic, but not so easy to interpret because there are many terms. The approach used here is to approximate three interesting, physical limits analytically to obtain some insight. Quantitative results are then presented by solving the equation numerically, subject to appropriate constraints.

## 6. Analytic limits

The task now is to examine the ionisation instability criterion and determine, or at least gain some insight on, how it impacts MHD generator performance. In principle, this could be accomplished by designing an MHD generator, varying several key parameters and learning how a figure of merit, for instance defined as the fraction of ‘furnace’ enthalpy converted to load power, is maximised when subject to the instability constraint. This is not possible with the present analysis because the stability criterion is a local one. In other words, the local criterion by itself does not determine the generator length, cross-sectional area and global performance.

What is needed instead is an alternate local figure of merit that intuitively measures generator desirability. Two plausible choices are as follows. For the first, assume the stability criterion is satisfied locally at each axial location of the generator by appropriately shaping the cross-section using advanced 3-D manufacturing techniques. A certain fraction of the ‘furnace’ kinetic plus thermal energy is converted to electricity. Of this converted electricity, part provides Ohmic heating ($S_{\Omega }$) of the electrons and the remainder is the desired power delivered to the load ($S_{L}$). Intuitively, one wants most of the converted electricity going to the load rather than heating the electrons. Thus, the ratio

**Equation (6.1)**

\begin{equation} \frac{S_{\Omega }}{S_{L}}=\frac{\text{Ohmic power density}}{\text{Load power density}} \end{equation}

evaluated at the generator inlet should be a reasonable figure of merit describing generator performance. Obviously, one wants $S_{\Omega }/S_{L}$ as small as possible for a desirable generator.

The second figure of merit is the electrical conductivity of the weakly ionised plasma,

**Equation (6.2)**

\begin{equation} \sigma =1/\eta =\frac{n_{e}e^{2}}{m_{e}\nu _{M}}, \end{equation}

again evaluated at the inlet. Intuitively, a high electrical conductivity is desirable for large currents and corresponding high power densities to be generated. This too is a plausible measure of generator performance.

To gain some insight, the stability criterion is analytically simplified for three cases of interest to see what the resulting impact is on generator performance as determined by the two figures of merit just defined. Also examined is whether or not high field (i.e. high $\beta$) helps or hurts the situation. Intuitively, high field should be useful since it leads to a large Hall voltage and corresponding high power to the load. However, the stability criterion places a limit on the maximum allowable field. The question is which of these plays the stronger role. The three cases analysed are as follows: (i) open cycle fossil fuel generator; (ii) standard closed cycle generator in which the seed gas is only weakly ionised; and (iii) advanced closed cycle generator with near full ionisation. Does the analysis show that near full ionisation leads to the attractive performance that has been observed experimentally?

We shall see that to carry out the analysis for each case considered, it is necessary to estimate the size – small, medium or large – of the quantities $\Delta T,\,\beta,\,f_{I},\,Z$ from which one can then deduce the sizes of $S_{\Omega }/S_{L}$ and $\sigma$.

### 6.1. Open cycle fossil fuel generator

An open cycle fossil fuel generator is characterised by a weakly ionised seed gas that is at nearly the same temperature as the primary gas. To obtain a high electrical conductivity, the temperature must be relatively high ($T_{e}\approx T_{p}\sim 2500\,\mathrm{K}$). The need for a high temperature can lead to material problems, which are one of the drawbacks of an open cycle fossil fuel system.

Keep in mind that the analysis so far has focused on closed cycle systems, which assume a monatomic gas. For a fossil fuel system, essentially all of the analysis still applies, with one exception. The relation between $\nu _{E}$ and $\nu _{M}$ is modified as follows:

**Equation (6.3)**

\begin{equation} \nu _{E}=2\frac{m_{e}}{m_{p}}\nu _{M}\rightarrow \nu _{E}=2\delta \frac{m_{e}}{m_{p}}\nu _{M}.\end{equation}

Here, $\delta$ is an energy equilibration enhancement factor whose value depends on the primary gas used in the generator (see for instance Rosa (1987)). For a closed cycle system using a monatomic gas (e.g. Ar), then $\delta =1$. For an open cycle system using a mixture of complex molecular gases (e.g. CO, CO2, NO), then $\delta > 200$. When this effect is included in the analysis, the important place in which it appears is in the relationship between $\Delta T$ and $Z$. Specifically, this relationship, given by (3.14), is now replaced by

**Equation (6.4)**

\begin{equation} \Delta T=\left(\frac{5M^{2}}{9}\right)\frac{\beta ^{2}\left[\beta ^{2}+\left(1+Z\right)^{2}\right]}{\left(\beta ^{2}+1+Z\right)^{2}}\rightarrow \Delta T=\left(\frac{5M^{2}}{9\delta }\right)\frac{\beta ^{2}\left[\beta ^{2}+\left(1+Z\right)^{2}\right]}{\left(\beta ^{2}+1+Z\right)^{2}}. \end{equation}

With this modification, consider the limit of an open cycle fossil fuel plant defined by $f_{I}\ll 1$ and $\delta \gg 1$. The appropriate orderings and approximations are as follows. First, the assumption of a low seed gas ionisation fraction $f_{I}\ll 1$, implies that

**Equation (6.5)**

\begin{equation} \alpha =\frac{kT_{e}}{2E_{I}}\frac{2-f_{I}}{1-f_{I}}\approx \frac{kT_{e}}{E_{I}}\ll 1. \end{equation}

Second, the energy balance relation between $\Delta T$ and $\beta,\,Z$, given in (6.4), shows that for modest to high Hall parameters, $\beta \sim 1$ to $\beta \gg 1$, and typical Mach numbers $M\sim 1{-}2$, the temperature difference is small for any $Z\boldsymbol{\lesssim }\beta$,

**Equation (6.6)**

\begin{equation} \Delta T=\left(\frac{5M^{2}}{9\delta }\right)\frac{\beta ^{2}\left[\beta ^{2}+\left(1+Z\right)^{2}\right]}{\left(\beta ^{2}+1+Z\right)^{2}}\sim \frac{1}{\delta }\ll 1. \end{equation}

The temperature difference is small as expected in an open cycle system with a complex molecular primary gas – energy transfer between electrons and primary gas is rapid when $\delta \gg 1$.

Next, to apply the stability criterion, one needs to estimate $\alpha /\Delta T$, the ratio of two small parameters. Since $\alpha \sim 1/20$ and $\delta \boldsymbol{>rsim }200$, observe that

**Equation (6.7)**

\begin{equation} \frac{\alpha }{\Delta T}\sim \alpha \delta \gg 1. \end{equation}

This scaling is substituted into the expression for the stability $\beta$ limit leading to the simplified result,

**Equation (6.8)**

\begin{align} \beta ^{2}&\leq 4\alpha \left(2+\frac{1}{\Delta T}\right)\left[1+\alpha \left(1+\frac{1}{\Delta T}\right)\right]\nonumber\\ &\approx \left(\frac{2\alpha }{\Delta T}\right)^{2}\gg 1\rightarrow \beta \leq \frac{2\alpha }{\Delta T}. \end{align}

While the instability limits the value of $\beta$, this limit is high; that is, it does not pose a serious constraint.

All quantities, except $Z$ have now been scaled. This last scaling is obtained by substituting into the first figure of merit and then minimising with respect to $Z$. From (6.1), one sees that the figure of merit reduces to

**Equation (6.9)**

\begin{equation} \frac{S_{\Omega }}{S_{L}}=\frac{\left[\beta ^{2}+\left(1+Z\right)^{2}\right]}{\beta ^{2}Z}. \end{equation}

This function has a minimum with respect to $Z$ at

**Equation (6.10)**

\begin{equation} Z=\left(1+\beta ^{2}\right)^{1/2}\sim \beta, \end{equation}

which is consistent with the scaling assumed previously, and results in a figure of merit given by

**Equation (6.11)**

\begin{equation} \frac{S_{\Omega }}{S_{L}}=2\frac{1+\left(1+\beta ^{2}\right)^{1/2}}{\beta ^{2}}\sim \frac{1}{\beta }. \end{equation}

High $\beta$, or equivalently high $B_{0}$, leads to good performance.

Turning to the second figure of merit, one sees that it scales as

**Equation (6.12)**

\begin{equation} \sigma =\frac{e^{2}n_{e}}{m_{e}\nu _{M}}\sim \frac{n_{e}}{T_{e}^{1/2}}.\end{equation}

This form is misleading in that raising $n_{e}$ and lowering $T_{e}\approx T_{p}$ to increase $\sigma$ also increases the contributions of $\nu _{en}$ and $\nu _{ei}$ to $\nu _{M}$. When these contributions are included, it has been shown (Rosa 1987) that there is actually an optimum density that maximises $\sigma$. At the optimum, $n_{e}(T_{e})$ and $\sigma (T_{e})$ are given by

**Equation (6.13)**

\begin{align}n_{e}^{2}\left(T_{e}\right)&=\left(\frac{p_{p}\overline{\sigma }_{ep}}{\overline{\sigma }_{en}}\right)\left[\frac{N(T_{e})}{kT_{e}}\right],\nonumber\\[4pt] \sigma(T_{e})&=\frac{e^{2}n_{e}}{m_{e}}\frac{1}{\left[2(\nu _{ep}\nu _{en})^{1/2}+\nu _{ei}\right]}.\end{align}

The conductivity $\sigma _{e}$ is a rapidly increasing function of $T_{e}$ and independent of $B_{0}$.

What are the main conclusions for an open cycle generator? There are three. (i) The ionisation instability poses a limit on the maximum allowable $\beta$, but this limit is high and thus does not represent a serious constraint on performance. (ii) The first figure of merit $S_{\Omega }/S_{L}$ becomes smaller as $\beta$ increases. In other words, high $\beta$, corresponding to high $B_{0}$, is a good strategy to maximise performance. (iii) The second figure of merit $\sigma$ is independent of $B_{0}$, but increases with $T_{e}\approx T_{p}$. High temperatures, of the order of 2500 K, are achievable with fossil fuel power plants.

### 6.2. Standard closed cycle generator

The open cycle fossil fuel generator has to deal with high temperature materials problems, which, while difficult, are not insurmountable. Even so, there is less interest now in fossil fuel plants, particularly those powered by coal, because of $\mathrm{CO}_{2}$ emissions.

In contrast, because of basic physics and engineering constraints, nuclear power plants, both fission and fusion, cannot at present achieve comparably high temperatures in the primary coolant. Typically, $T_{p}\sim 1000{-}1300\,\mathrm{K}$ is closer to the upper limit. Because of the strong exponential $T_{e}$ dependence in the Saha equation, the electron density would be reduced by approximately 10 orders of magnitude when $T_{e}\approx T_{p}$ is reduced by a factor of approximately 2, from 2000 K to 1000 K! This has motivated the idea of using a closed cycle generator, with a monatomic primary gas such as argon (Kerrebrock 1964; Kerrebrock & Hoffman 1964; Sheindlin *et al*. 1964). The reasoning is that with the long energy equilibration time between argon and seed electrons, say potassium, plus the continuous Ohmic heating along the channel, it should be possible to maintain a finite temperature difference between the two species. ‘We can have our cake and eat it’. Lower temperature argon, easy to produce in a nuclear plant, combined with high temperature electrons needed for large electrical conductivity, should lead to highly efficient energy conversion.

Unfortunately, here is where the ionisation instability enters the picture. This instability is excited when there is a substantial temperature difference. The question then is how much impact does the maximum allowable stable temperature difference have on the energy conversion efficiency, as defined by the figures of merit? This question is now addressed for a standard closed cycle MHD generator defined by $f_{I}\ll 1$ and $\delta =1$. The analysis proceeds as follows.

As noted previously, the weakly ionised seed gas assumption, $f_{I}\ll 1$, leads to the ordering

**Equation (6.14)**

\begin{equation} \alpha =\frac{kT_{e}}{2E_{I}}\frac{2-f_{I}}{1-f_{I}}\approx \frac{kT_{e}}{E_{I}}\ll 1.\end{equation}

Now, since the primary gas temperature is assumed to be low, one needs to operate in a regime where the temperature difference is no longer small, but is instead finite: $\Delta T\sim 1$. However, this assumption leads to a contradiction. Specifically, for $\Delta T\sim 1$, the stability condition, (6.8), requires that $\beta \sim \alpha ^{1/2}\ll 1$. When substituted into the energy balance relation, (6.6), and setting $\delta =1$ for a monatomic gas, one sees that this implies $\Delta T\sim \beta ^{2}\ll 1$, which violates the original assumption.

After some thought, note that for a weakly ionised seed gas in a monatomic primary gas, the self-consistent ordering assumption becomes

**Equation (6.15)**

\begin{align} \alpha &\ll 1,\nonumber\\ \Delta T\sim \alpha ^{1/2}&\ll 1,\nonumber\\ \alpha /\Delta T\sim \alpha ^{1/2}&\ll 1. \end{align}

These orderings imply that

**Equation (6.16)**

\begin{align} \Delta T&\approx \frac{5M^{2}}{9}\beta ^{2}\sim \alpha ^{1/2}\ll 1&&\quad \text{Energy balance},\nonumber\\ \beta ^{2}&\leq \frac{4\alpha }{\Delta T}\sim \alpha ^{1/2}\ll 1&&\quad \text{Stability condition}. \end{align}

The two figures of merit reduce to

**Equation (6.17)**

\begin{align} \frac{S_{\Omega }}{S_{L}}&\approx \frac{\left(1+Z\right)^{2}}{\beta ^{2}Z}\approx \frac{4}{\beta ^{2}}\sim \frac{1}{\alpha ^{1/2}}\gg 1&&\quad \text{Minimized figure of merit}\,(Z=1),\nonumber\\ \sigma \left(T_{e}\right)&=\frac{e^{2}n_{e}}{m_{e}}\frac{1}{\left[2\left(\nu _{ep}\nu _{en}\right)^{1/2}+\nu _{ei}\right]}&&\quad T_{e}\approx T_{p}. \end{align}

This corresponds to an unattractive mode of operation: (i) the temperature difference is small because of the instability (making it difficult to obtain a high $\sigma$); (ii) the Hall parameter is small (leading to an undesirably low Hall voltage); (iii) $S_{\Omega }/S_{L}$ is large (indicating that most of the converted electricity is going into Ohmic heating and not the load); and (iv) the conductivity is very small because of the low electron density implied by the low temperature in Saha’s equation.

The main conclusion from this analysis is that the ionisation instability imposes a strong constraint on the performance of a closed cycle MHD generator using a monatomic primary gas plus a weakly ionised seed current. This conclusion was borne out experimentally early in the programme (Velikhov & Dykhne 1963; Velikhov *et al*. 1965). The instability has largely been viewed as a ‘show stopper’ and much of the research in this area has been strongly curtailed.

### 6.3. Advanced closed cycle generator

In spite of the dire theoretical predictions and poor experimental performance, experimentalists did find a way to improve closed cycle operation. It was observed that when the seed gas became essentially fully ionised, the ionisation instability would be suppressed. There have been several contributions in the theoretical literature (Petit & Valensi 1969; Mitchner & Kruger 1973; Nakamura & Riedmuller 1974; Kien 2016) that support this conclusion but, to the author’s knowledge, no sharply defined stability condition and corresponding scaling relations have explicitly appeared. This gap is filled in the present work and represents one important contribution. It is also worth noting that while performance could be improved experimentally, the gains were not sufficiently large so as to strongly regenerate interest in this area of research. A second important contribution of the present work is to investigate whether access to much higher magnetic fields, now possible because of REBCO superconductors, can lead to much larger gains in performance, perhaps sufficiently large to regenerate interest in closed cycle MHD energy conversion.

The advanced closed cycle MHD generator is defined by $f_{I}\rightarrow 1$ and $\delta =1$. The analysis to predict performance again requires some thought with respect to the orderings of various quantities and the corresponding experimental consequences. The starting point is the assumption of near full ionisation of the seed gas: $f_{I}\rightarrow 1$. In fact, for best performance, the seed gas must be very near full ionisation so that the following ordering holds for the ionisation parameter:

**Equation (6.18)**

\begin{equation} \alpha =\frac{kT_{e}}{2E_{I}}\frac{2-f_{I}}{1-f_{I}}\approx \frac{kT_{e}}{2E_{I}}\frac{1}{1-f_{I}}\gg 1. \end{equation}

The parameter has obviously switched from being very small to very large, which is the key mathematical insight needed to produce a high performance closed line MHD generator.

Best performance also requires operation in the regime of a large Hall parameter plus a large load impedance. Not so obviously, the relative size of these two parameters must satisfy

**Equation (6.19)**

\begin{equation} \beta ^{2}\gg Z\gg \beta \gg 1. \end{equation}

For example, $Z\sim \beta ^{3/2}$. When this assumption is substituted into the energy equation, this leads to

**Equation (6.20)**

\begin{equation} \Delta T=\left(\frac{5M^{2}}{9}\right)\frac{\beta ^{2}\left[\beta ^{2}+\left(1+Z\right)^{2}\right]}{\left(\beta ^{2}+1+Z\right)^{2}}\approx \left(\frac{5M^{2}}{9}\right)\frac{Z^{2}}{\beta ^{2}}\gg 1. \end{equation}

The temperature difference is now large, one desirable goal. In the interesting operational regime, the ratio of the two large parameters $\alpha,\,\Delta T$ is assumed to satisfy

**Equation (6.21)**

\begin{equation} \frac{\alpha }{\Delta T}\approx \left(\frac{9kT_{e}}{10E_{I}M^{2}}\right)\frac{\beta ^{2}}{Z^{2}}\frac{1}{1-f_{I}}\gg 1,\end{equation}

which sets the required level of ionisation. Next, substitution into the stability criterion leads to

**Equation (6.22)**

\begin{align} \beta ^{2} & \leq 4\alpha \left(2+\frac{1}{\Delta T}\right)\left[1+\alpha \left(1+\frac{1}{\Delta T}\right)\right]\nonumber\\[3pt]& \approx 8\alpha ^{2}\gg 1, \end{align}

which reduces to

**Equation (6.23)**

\begin{equation} \beta \leq 2^{1/2}\frac{kT_{e}}{E_{I}}\frac{1}{1-f_{I}}. \end{equation}

Equation (6.23) is an important result. It clearly shows the relation between the maximum stable $\beta$ and the degree of ionisation $1-f_{I}$. Stability at high $\beta$ requires nearly full ionisation.

Before proceeding, it is worthwhile noting that so far, it has not been necessary to specify a precise scaling of $Z$ with $\beta$, only the range given by (6.19). As such, a simple way to clarify the abovementioned tangle of orderings is to actually make two specific assumptions with respect to $Z\text{ and }kT_{e}/E_{I}: Z\sim \beta ^{\kappa }$ with $1< \kappa < 3/2$ and $kT_{e}/E_{I}\sim 1/\beta$, both consistent with the abovementioned assumptions. Then, all relevant quantities can be scaled directly with $\beta$, and simultaneously satisfy all the orderings previously discussed,

**Equation (6.24)**

\begin{align} \frac{kT_{e}}{E_{I}} & \sim \frac{1}{\beta }\ll 1,\nonumber\\ Z & \sim \beta ^{\kappa }\gg 1 \quad \textrm{and}\quad 1< \kappa < 3/2,\nonumber\\ \Delta T & \sim \beta ^{2\left(\kappa -1\right)}\gg 1,\nonumber\\ \alpha & \sim \beta \gg 1,\nonumber\\ \frac{\alpha }{\Delta T} & \sim \beta ^{3-2\kappa }\gg 1,\nonumber\\ 1-f_{I} & \sim \frac{1}{\beta ^{2}}\ll 1. \end{align}

We can now use this ordering scheme to evaluate the two figures of merit. Consider first $S_{\Omega }/S_{L}$. A short calculation yields

**Equation (6.25)**

\begin{equation} \frac{S_{\Omega }}{S_{L}}=\frac{\left[\beta ^{2}+\left(1+Z\right)^{2}\right]}{\beta ^{2}Z}\approx \frac{Z}{\beta ^{2}}\sim \frac{1}{\beta ^{2-\kappa }}\ll 1. \end{equation}

If one wants to minimise $S_{\Omega }/S_{L}$, then high $\beta$ (i.e. high $B_{0}$) is a good strategy.

The second figure of merit requires a little more work, which makes use of the relations $n_{e}\approx n_{s}, 1-f_{I}=n_{s}/N\approx n_{e}/N$ and $\beta =(eB_{0}/m_{e})/\nu _{M}=\Omega _{e}/\nu _{M}$. The result is

**Equation (6.26)**

\begin{equation} \sigma =\frac{e^{2}n_{e}}{m_{e}\nu _{M}}=\left(\frac{e^{2}n_{e}}{m_{e}}\right)\left(\frac{NkT_{e}}{2E_{I}\alpha }\right)\leq 2^{1/2}\left(\frac{kT_{e}}{E_{I}}\right)\left(\frac{e^{2}N}{m_{e}\Omega _{e}}\right). \end{equation}

We see that if the goal is to achieve large $\sigma$, then small $\Omega _{e}$ (i.e. small $B_{0}$) is the path to take.

As stated earlier, the two figures of merit have opposite requirements on $B_{0}$ to achieve good performance. Low $S_{\Omega }/S_{L}$ requires high $B_{0}$, while high $\sigma$ requires low $B_{0}$. The physical explanation for these opposing requirements is as follows. The conductivity is proportional to $n_{e}$. Now, as $B_{0}$ increases, the need to ever more closely approach full ionisation for stability requires, by virtue of Saha’s equation, that the seed density $n_{s}$ decreases. Since $n_{e}\approx n_{s}$, the implication is the electron density and, hence, the conductivity will also decrease as $B_{0}$ increases. In contrast, for the second figure of merit $S_{\Omega }/S_{L}$, the electron density cancels when forming the ratio. The resulting $S_{\Omega }/S_{L}$ is only a function of $\beta \propto B_{0}/T_{e}^{1/2}$ with the $B_{0}$ dependence dominating, since the electron temperature only varies slightly because of its exponential dependence in Saha’s equation. The high performance associated with high field leads to the result that $S_{\Omega }/S_{L}$ will decrease as $B_{0}$ increases, which is a favourable result.

Resolving this dichotomy requires a global solution to MHD generator design, which is not possible using only a local stability criterion. The global analysis has been completed and will be reported on in the near future. As a preview, note that a global design is actually strongly influenced by engineering constraints which are not considered here. Including these constraints demonstrates that there is a high but optimum magnetic field that maximises performance.

We close this subsection by discussing one further major point – the significance of the precise scaling of the ionisation fraction $f_{I}$. After all, it seems extremely difficult to measure or control the ionisation so accurately. This, however, is not the main concern. Instead, the nearness to full ionisation provides a strong constraint between the electron temperature and electron density via Saha’s equation. The constraint leads to the following relation:

**Equation (6.27)**

\begin{equation} n_{e}\left(T_{e}\right)=\left(\frac{kT_{e}}{2E_{I}}\right)\left(\frac{N}{\alpha }\right)\leq 2^{1/2}\left(\frac{kT_{e}}{2E_{I}}\right)\left(\frac{N}{\beta }\right), \end{equation}

which sets the maximum allowable density of the unionised seed gas. This is a critical design constraint for closed cycle MHD generators to be used as a topping cycle for a nuclear power plant.

What are the conclusions with respect to advanced closed cycle MHD Hall generators? Overall, they are positive: (i) the temperature difference is large, which is just what is needed to create a high conductivity plasma assuming a low primary gas temperature; (ii) the seed gas must be very close to full ionisation for stability, as observed experimentally; and (iii) the two figures of merit make conflicting predictions as to the value of high field. This is not necessarily bad, but more work on global designs is needed to resolve the conflict and determine the optimum magnetic field.

## 7. Numerical results

The last section in the paper focuses on the nearly full ionisation, closed cycle MHD generator. Here, the exact, unexpanded stability equations are solved numerically, using a set of practical numerical values, to obtain a reasonably quantitative picture of generator performance in un-normalised units. It is also verified that the analytic scaling relations discussed previously are well satisfied.

The analysis is slightly complicated by the fact that when switching to real units, there is a critical missing piece of information. Ideally, when actually designing an MHD topping cycle generator, one needs a well-defined global technical goal, for instance, to achieve an enthalpy conversion efficiency of 35 %. Note that this is equivalent to 35 % of the total input enthalpy being converted directly into electricity in the load. When the remaining 65 % of the enthalpy is fed into a standard steam bottoming cycle, the overall plant efficiency is approximately 55 %.

Unfortunately, this type of non-local goal is not possible to implement using only a local theory of stability. Instead, a local replacement for the global goal is needed. A good strategy is to assume that the marginal stability criterion is satisfied locally along the entire generator length, and then choose a meaningful physical quantity at the inlet as an alternate local goal. One possible, although not unique, choice for this quantity is the total converted electric power density, $S_{C}$. This quantity strongly impacts economics. Typically, $S_{C}$ in other types of power generators may be of the order of $100\text{ MW}\,\mathrm{m}^{-3}$. The following results assume that the generator under consideration has a specified value of $S_{C}$ at the inlet, and for comparison, three different values, $S_{C}=100,200,300\,\mathrm{MW}\,\mathrm{m}^{-3}$, are considered.

Also, when substituting numerical values, a high temperature gas cooled fission reactor (HTGR) is chosen as the ‘furnace’ (NEA 2022). The HTGR uses helium as the coolant. The nuclear heated helium passes through an energy exchanger in which the secondary gas is argon. The argon itself is then passed through a Laval nozzle to produce a gas flow with a Mach number greater than unity, which is important in achieving high energy transfer efficiency in a Hall generator.

Our goal is to substitute practical values for the quantities of interest and then calculate, using the previous analysis, the figures of merit, $S_{\Omega }/S_{L}$ and $\sigma$, plus other quantities of physical interest, as a function of the magnetic field $B_{0}$. Does high field offer the possibility of a substantially improved MHD energy convertor?

### 7.1. Input parameters

The first step in the analysis is to choose ‘furnace’ values corresponding to the Argon gas leaving the heat exchanger. These values are held fixed during all calculations. For the HTGR they are given by (NEA 2022):

(a) argon coolant input pressure to the Laval nozzle: $p_{in}=5\times 10^{6}\text{ Pa}\approx 50\text{ atm}$;

(b) argon coolant input temperature to the Laval nozzle: $T_{in}=1000\,\mathrm{K}$;

(c) argon coolant input velocity to the Laval nozzle: highly subsonic $v_{in}^{2}\ll 2kT_{in}/m_{p}$;

(d) argon coolant output Mach number from the Laval nozzle: $M=1.8$.

The output properties of the Laval nozzle gas serve as the primary argon input quantities to the MHD generator.

\begin{align*} &\quad\!\!\left(\mathrm{a}\right)\text{Argon MHD inlet pressure}{\colon} \nonumber\\ &\qquad p_{p}=\left(\!1+\frac{\gamma -1}{2}M^2\!\right)^{-\frac{\gamma }{\gamma -1}}p_{in}=\left(\frac{3}{M^{2}+3}\right)^{5/2}p_{in}=0.801\times 10^{6}\text{ Pa}\approx 8.01\text{ atm}. \end{align*}

(b) Argon MHD inlet temperature:

\begin{align*} T_{p}=\left(\!1+\frac{\gamma -1}{2}M^2\!\right)^{-1}T_{in}=\left(\frac{3}{M^{2}+3}\right)T_{in}=481\,\mathrm{K}. \end{align*}

(c) Argon MHD inlet number density:

\begin{align*}n_{p}=\frac{p_{p}}{kT_{p}}=\left(1+\frac{\gamma -1}{2}M^2\right)^{-\frac{1}{\gamma -1}}\frac{p_{in}}{kT_{in}}=\left(\frac{3}{M^{2}+3}\right)^{3/2}\frac{p_{in}}{kT_{in}}=1.21\times 10^{26}\,\mathrm{m}^{-3}. \end{align*}

(d) Argon MHD inlet velocity:

\begin{align*} v_{p}=\left(\frac{\gamma kT_{p}M^{2}}{m_{p}}\right)^{1/2}=\left(\frac{5kT_{p}M^{2}}{3m_{p}}\right)^{1/2}=735\,\mathrm{m}\,\sec^{-1}. \end{align*}

Here, $\gamma$ is the ratio of specific heats (for a monatomic gas, $\gamma =5/3$). The abovementioned values are held fixed for all calculations. Observe that the MHD generator input temperature is 481 K, substantially reduced from the initial 1000 K because of the need for a Laval nozzle to produce a supersonic flow velocity.

Two other input quantities that are held fixed during a given calculation are the magnetic field and converted electric power density. Separate calculations scan the values of these two quantities, whose range covers:

(i) Magnetic field: $3\,\mathrm{T}< B_{0}< 20\,\mathrm{T}$;

(ii) Power density: $S_{C}=100,200,300\,\mathrm{MW}\,\mathrm{m}^{-3}$.

The magnetic field is allowed to vary over a wide range. The lower value of 3 T corresponds to typical operation of MHD generators prior to the almost complete termination of the USA programme in the 1990s. The higher value of 20 T is at the limit of practical magnetic fields using the recently developed REBCO superconductors (Vieira *et al.* 2024). As discussed, the second scanning parameter of interest is the inlet converted electric power density and three plausible discrete values are chosen. Too low a value outside this range implies a large, uneconomical generator. Too high a value translates into serious material problems on the generator walls and electrodes.

### 7.2. How to obtain a solution

Assume now that all the input parameters have been specified, including values for $B_{0}$ and $S_{C}$. It is shown now that obtaining values for all the physical quantities of interest requires the solution to a nonlinear algebraic equation for the temperature $T_{e}$, a simple numerical calculation. The procedure requires making an initial guess for $T_{e}$ and then evaluating the following quantities in the order listed,

\begin{align*} &\Delta T=\frac{T_{e}}{T_{p}}-1&&\quad \text{Definition}\nonumber\\ &\nu_{M}=n_{p}\overline{\sigma }_{ep}\left(\frac{2kT_{e}}{m_{e}}\right)^{1/2}&&\quad \text{Definition},\nonumber\\ &\beta =\frac{eB_{0}}{m_{e}\nu _{M}}&&\quad \text{Definition},\nonumber\\ &Z=\xi +\left[\frac{\xi \left(\xi +1\right)}{\mu }\right]^{1/2}&&\quad \text{Energy balance},\nonumber\\ &\mu =\frac{9}{5M^{2}\beta ^{2}}\Delta T &&\quad \text{Energy balance},\nonumber\\ &\xi =\frac{\beta ^{2}\mu }{1-\mu }-1&&\quad \text{Energy balance},\nonumber \end{align*}

**Equation (7.1)**

\begin{align} &n_{e}=\frac{\left(\beta ^{2}+1+Z\right)}{\beta ^{2}\left(1+Z\right)}\frac{S_{C}}{m_{e}\nu _{M}v_{p}^{2}}&&\qquad\qquad \text{Power density},\nonumber\\ &1-f_{I}=\frac{n_{e}/N\left(T_{e}\right)}{1+n_{e}/N\left(T_{e}\right)}&&\qquad\qquad \text{Saha},\nonumber\\ &\alpha =\frac{1}{2}\left(\frac{2kT_{e}}{3kT_{e}+2E_{I}}\right)\left(\frac{2-f_{I}}{1-f_{I}}\right)&&\qquad\qquad \text{Definition}.\end{align}

For algebraic simplicity, $\gamma =5/3$ has substituted wherever appropriate in these expressions.

All the quantities of interest have now been evaluated for the given guess of $T_{e}$. These values are now substituted into the marginal stability criterion for the ionisation instability, repeated here for convenience,

**Equation (7.2)**

\begin{equation} \beta ^{2}=4\alpha \left(2+\frac{1}{\Delta T}\right)\left[1+\alpha \left(1+\frac{1}{\Delta T}\right)\right]. \end{equation}

In general, this constraint will not be satisfied for the $T_{e}$ guess. It is here that a simple numerical iteration on $T_{e}$ is all that is required to satisfy the marginal stability constraint.

### 7.3. Results

Following the procedure just discussed, a large number of cases have been evaluated. The most important results are summarised in figures 2 and 3, where the figures of merit $S_{\Omega }/S_{L}$ and $\sigma$ have been plotted versus magnetic field $B_{0}$ for three values of $S_{C}$.

> Figure 2. Figure of merit $S_{\Omega }/S_{L}$ versus magnetic field for three values of $S_{C}\,(\mathrm{MW}\,\mathrm{m}^{-3})$.

> Figure 3. Figure of merit $\sigma$ versus magnetic field for three values of $S_{C}\,(\mathrm{MW}\,\mathrm{m}^{-3})$.

Several conclusions can be drawn. First, $S_{\Omega }/S_{L}$ decreases rapidly with magnetic field, transforming from undesirable values greater than unity to attractive values much less than unity. In other words, high field is a potential winner for improving the efficiency of closed cycle MHD generators operating at the marginal stability boundary of the ionisation instability. Second, the figure of merit is almost independent of the required load power density over a reasonably wide range $100< S_{C}(\mathrm{MW}\,\mathrm{m}^{-3})< 300$. This is not surprising since the scaling factor $S_{C}$ cancels when calculating the ratio $S_{\Omega }/S_{L}$. Lastly, as shown by the dashed line, to keep $S_{\Omega }/S_{L}$ below 20 % requires a magnetic field of 14 T or greater. The 20 % value is a desirable practical goal implying that 80 % of the converted electrical power is supplied to the load and only 20 % to heat the electrons.

The conclusions with respect to the second figure of merit $\sigma$, as shown in figure 3, are quite the opposite. Here, lower magnetic field leads to higher conductivity. Also, at any given magnetic field, the conductivity is approximately linearly proportional to the value of $S_{C}$. Higher conductivity requires a higher electron density which, in turn, requires a higher converted power density. Also, in almost all cases, the conductivity is relatively high, greater than $10\text{ mho}\,\mathrm{m}^{-1}$, which is the requirement for a high-quality plasma even though the primary gas temperature is only 481 K.

Additional useful information is illustrated in figure 4. Plotted here are (*a*) electron temperature $T_{e}$, (*b*) electron density $n_{e}$, (*c*) Hall parameter $\beta$ and (*d*) the fraction of unionised seed gas $1-f_{I}$, all versus the magnetic field $B_{0}$.

> Figure 4. Curves of (*a*) $n_{e}$, (*b*) $T_{e}$, (*c*) $\beta$ and (*d*) $1-f_{I}$, versus $B_{0}$ for three values of $S_{C}$.

The following points are worth noting. The required temperature is, interestingly, almost independent of magnetic field and load power density. The reason can be traced back to the strong exponential dependence in the Saha equation. Even small changes in the electron temperature result in enormous changes in the electron density, which would lead to comparably large changes in the load power density.

The electron density decreases rapidly as the field increases. This is a consequence of needing ever lower densities to more closely approach full ionisation, as required by the marginal stability criterion. The density is also a rapidly increasing function of the power density to the load. This is not surprising – more load power requires more current, which in turn requires more electrons.

The Hall parameter increases linearly with $B_{0}$, which is to be expected since $\beta \propto B_{0}$. Values of $\beta$ of the order of 10–20, much larger than those in early experiments, are needed if high field is desirable. Also, the seed gas becomes progressively closer to full ionisation as $B_{0}$ increases. As stated, this is a consequence of the ionisation instability marginal criterion. Both quantities are almost independent of $S_{C}$.

As a specific reference case, assume values for the critical generator parameters given by $M=1.8, S_{C}=100\text{ MW}\,\mathrm{m}^{-3}$ and $B_{0}=8\mathrm{T}$. It then follows that

**Equation (7.3)**

\begin{align} S_{\Omega }/S_{L}&=0.3720,\nonumber\\ \sigma &=12.35\text{ mho}\,\mathrm{m}^{-1},\nonumber\\ T_{e}&=4223\,\mathrm{K},\nonumber\\ T_{e}/T_{p}&=8.783,\nonumber\\ n_{e}&=7.478\times 10^{19}\mathrm{m}^{-3},\nonumber\\ n_{e}/n_{p}&=6.180\times 10^{-7},\nonumber\\ \beta &=8.249\nonumber\\ 1-f_{I}&=0.01679.\end{align}

As compared with earlier, lower field generators, note the higher temperature ratio and higher Hall parameter.

The last point of interest involves a comparison of results for two other forms of the marginal stability boundary. Keep in mind that the abovementioned results correspond to an exact numerical solution of the stability criterion, but using the approximate form of collision frequency $\nu _{M}\approx \nu _{ep}$. The first comparison involves the $\nu _{M}\approx \nu _{ep}$ approximation. As previously stated, but not proved, it has been assumed that the dominant collision mechanism is between electrons and the primary gas. One can now offer proof by recalculating the results presented in figures 2 and 3, i.e. the curves of $S_{\Omega }/S_{L}\text{ vs }B_{0}$ and $\sigma \text{ vs }B_{0}$ for fixed $S_{C}=100\text{ MW}\,\mathrm{m}^{-3}$. The new curves are obtained by setting (see for instance Wesson 2011 for $\nu _{ei}$)

**Equation (7.4)**

\begin{align} \nu _{M}&=\nu _{ep}+\nu _{en}+\nu _{ei},\nonumber\\[4pt] \nu _{ep}&=n_{p}\overline{\sigma }_{ep}v_{Te},\nonumber\\[4pt] \nu _{en}&=\left(n_{s}-n_{e}\right)\overline{\sigma }_{en}v_{Te},\nonumber\\[4pt] \nu _{ei}&=\frac{2^{1/2}}{12\pi ^{3/2}}\frac{n_{e}e^{4}ln\Lambda }{\varepsilon _{0}^{2}m_{e}^{1/2}\left(kT_{e}\right)^{3/2}}\quad\Lambda =4\pi \frac{\varepsilon _{0}^{3/2}\left(kT_{e}\right)^{3/2}}{e^{3}n_{e}^{1/2}} \end{align}

and solving the equations numerically. Results are presented shortly.

The second comparison involves the analytic approximation for the stability boundary. Here, one again assumes $\nu _{M}\approx \nu _{ep}$ and uses the approximate form of the stability boundary given by (6.22), which leads to

**Equation (7.5)**

\begin{align} \frac{S_{\Omega }}{S_{L}}&\approx \frac{Z}{\beta ^{2}},\nonumber\\ \sigma &\approx 2^{1/2}\left(\frac{kT_{e}}{E_{I}}\right)\left(\frac{e^{2}N}{m_{e}\Omega _{e}}\right). \end{align}

Solutions are obtained using the procedure described in (7.1) and (7.2),

**Equation (7.6)**

\begin{align} &\Delta T\approx \frac{T_{e}}{T_{p}}&&\quad \text{Definition},\nonumber\\[3pt] &\nu _{M}\approx n_{p}\overline{\sigma }_{ep}\left(\frac{2kT_{e}}{m_{e}}\right)^{1/2}&&\quad \text{Definition},\nonumber\\[3pt] &\beta =\frac{eB_{0}}{m_{e}\nu _{M}}&&\quad \text{Definition},\nonumber\\[3pt] &Z\approx \left(\frac{9\Delta T}{5M^{2}}\right)^{1/2}\beta &&\quad \text{Energy balance},\nonumber\\[3pt] &n_{e}\approx \frac{S_{C}}{m_{e}\nu _{M}v_{p}^{2}Z}&&\quad \text{Power density},\nonumber\\[3pt] &1-f_{I}\approx n_{e}/N\left(T_{e}\right)&&\quad \text{Saha},\nonumber\\[3pt] &\alpha \approx \frac{kT_{e}}{2E_{I}}\left(\frac{1}{1-f_{I}}\right)&&\quad \text{Definition},\nonumber\\[3pt] &\beta ^{2}\approx 8\alpha ^{2}&&\quad \text{Marginal stability}. \end{align}

These approximations greatly simplify the analysis resulting in a simple algebraic equation for the electron temperature. After a slightly tedious calculation, one obtains

**Equation (7.7)**

\begin{align} w^{-7/2}\exp \left(-w\right)&=C,\nonumber\\[4pt] w&=\frac{E_{I}}{kT_{e}}, \nonumber\\[4pt] C&=\left[\frac{1}{\left(160\pi ^{3}\right)^{1/2}}\frac{m_{p}h^{3}k^{1/2}}{m_{e}^{2}E_{I}^{5/2}\overline{\sigma }_{ep}}\right]\left(\frac{T_{p}^{1/2}S_{C}}{Mp_{p}}\right)=7.715\times 10^{-13}\left(\frac{T_{p}^{1/2}S_{C}}{Mp_{p}}\right). \end{align}

Observe that the marginally stable electron temperature is independent of $B_{0}$. Also, because of the strong exponential dependence of $w$, the resulting electron temperature is only a weak function of $M,\,S_{C},\,T_{p},\,p_{p}$.

Assume now that $T_{e}$ (i.e. $w$) is known from a simple numerical solution of (7.7). This result is substituted into the expression for the figures of merit leading to

**Equation (7.8)**

\begin{align} \frac{S_{\Omega }}{S_{L}}&=\left[\left(\frac{18}{5}\right)^{1/2}\frac{m_{e}^{1/2}\overline{\sigma }_{ep}E_{I}}{ek^{3/2}}\right]\frac{p_{p}}{wT_{p}^{3/2}MB_{0}}=0.6050\frac{p_{p}}{wT_{p}^{3/2}MB_{0}},\nonumber\\[4pt] \sigma &=\left[4\pi ^{3/2}\frac{e\left(m_{e}E_{I}\right)^{3/2}}{h^{3}}\right]\frac{w^{-5/2}e^{-w}}{B_{0}}=6.184\frac{w^{-5/2}e^{-w}}{B_{0}}. \end{align}

Observe that $S_{\Omega }/S_{L}$ is inversely proportional to $B_{0}$. High field is desirable to maximise the fraction of converted power that is delivered to the load as opposed to heating electrons. The conductivity is also inversely proportional to $B_{0}$, but in this case, high field is undesirable. The conductivity decreases as the field increases.

The overall comparison of the three different theoretical models are illustrated in figures 5 and 6.

> Figure 5. Comparisons of $S_{\Omega }/S_{L}$ for three different stability models.

> Figure 6. Comparisons of $\sigma$ for three different stability models.

The basic stability model used to plot figures 2 and 3 is shown in blue. The more exact model, which includes electron-seed neutral and electron-ion collisions, is shown in orange. Both of these models are in good agreement for all magnetic fields, particularly at high field. The close comparison is the justification for focusing solely on electron–primary gas collisions in the analysis.

The analytic model is also in good agreement with both of the other models at high fields. It begins to deviate at lower fields where the assumption $\beta \gg 1$ starts to break down. The analytic model shows the linear inverse scaling of both figures of merit with $B_{0}$.

## 8. Summary

We have revisited the Velikhov-ionisation instability and its impact on closed cycle MHD energy generators to be used as a topping cycle for $\mathrm{CO}_{2}$-free nuclear power plants. For many years, the instability was viewed as a show stopper for this application, based on both theory and experiment. Still, some other experimental data, never fully exploited, indicated that operation near full ionisation of the seed gas could suppress the instability. Understanding this stabilisation is the basic problem that has been investigated.

One main contribution is a first-principles derivation that shows the direct connection between stabilisation of the mode and a high seed-ionisation fraction. Here, stabilisation refers to the largest allowable value of the Hall parameter $\beta$. A specific expression has been derived that predicts the maximum allowable seed density to achieve the desired high value of stable $\beta$. High values of $\beta$ imply a large ratio of $T_{e}/T_{p}$, of the order of 10, which is stable and very important in producing a highly conducting plasma when the primary gas is relatively cold. Such large ratios have not been observed in earlier experiments. It would indeed be interesting to carry out an experimental programme demonstrating these large stable values of $T_{e}/T_{p}$.

The second main contribution of the research has been an attempt to assess whether much higher magnetic fields, now accessible via REBCO superconductors, would help or hurt the performance of closed cycle MHD generators. The assessment has been made by examining the $B_{0}$ dependence of two locally defined figures of merit, the ratio of Ohmic to load power densities $S_{\Omega }/S_{L}$ and the electrical conductivity $\sigma$. Since the stability criterion is a local constraint, it is not mathematically possible to directly obtain more relevant global performance measures. Still, using local figures of merit should provide some valuable insight. The results are perhaps surprising in that $S_{\Omega }/S_{L}$ and $\sigma$ show an opposite desirability when increasing $B_{0}$. The ratio $S_{\Omega }/S_{L}$ improves with increasing $B_{0}$; that is, $S_{\Omega }/S_{L}$ gets smaller implying that a larger fraction of the converted electrical power goes to the load rather than Ohmic heating. However, increasing $B_{0}$ has a negative effect on $\sigma$; that is, $\sigma$ decreases as $B_{0}$ increases, indicating a poorer quality plasma. These conflicting scaling predictions represent a different balance between the (i) performance improvements associated with high $B_{0}$ and (ii) the stability improvement as $B_{0}$ decreases.

What is the bottom line? Overall, the access to higher temperature ratios by means of further increasing the ionisation towards $f_{I}\rightarrow 1$ is a good result, suggesting that high-quality plasmas are possible in closed cycle MHD generators for both high and low magnetic fields. The desirability of high field is undecided with the two figures of merit giving opposing conclusions. So, what is the answer to the question in the title of the paper ‘A New Opportunity for MHD Energy Conversion?’. The answer is ‘maybe’. New work has been completed that quantitatively addresses the global performance of MHD generators operating in the nearly fully ionised mode. Results will be presented in a future paper.

## Acknowledgements

The author would like to thank several scientists from MIT for very interesting and illuminating discussions concerning both fusion and fission power sources. They are Professor Dennis Whyte (Department of Nuclear Science and Engineering and the PSFC), Dr Samuel Frank (PSFC, currently at Realta Fusion) and Professor Koroush Shirvan (Department of Nuclear Science and Engineering).

*Editor Won Ho Choe thanks the referees for their advice in evaluating this article.*

## Funding

This work was partially supported by the Department of Energy– Fusion Energy Science Grant No. DE-FG02-91ER54109). The author would also like to gratefully acknowledge the funding provided by the MIT Energy Initiative’s Seed Fund Program.

## Declaration of interests

The author reports no conflict of interest.

## References

1. Hatori, S. & Shioda, S. 1974 Stabilization of the ionization instability in an MHD generator. J. Phys. Soc. Jpn. 36, 920 – 920. 10.1143/JPSJ.36.920 CrossRef
2. Karlovitz, B. 1940 Process for the conversion of energy. U.S. Patent No. 2,210,918.
3. Kerrebrock, J.L. 1964 Nonequilibrium ionization due to electron heating: I T heory. AAIA J. 2, 1072 – 1080. 10.2514/3.2496 CrossRef
4. Kerrebrock, J.L. & Hoffman, M.A. 1964 Nonequilibrium ionization due to electron heating: II Experiments. AAIA Journal 2, 1080 – 1087. 10.2514/3.2497 CrossRef
5. Kien, L.C. 2016 Analyses on the ionization instability of non-equilibrium seeded plasma in an MHD generator. Plasma Sci. Technol. 18, 674 – 679. 10.1088/1009-0630/18/6/15 CrossRef
6. Messerle, H.K. 1994 Magnetohydrodynamic Electric Power Generation. John Wiley & Sons.
7. Mitchner, M. & Kruger, C.H. 1973 Partially Ionize Gases. John Wiley & Sons.
8. Murakami, T., Okuno, Y. & Yamasaki, H. 2005 Suppression of ionization instability in a magnetohydrodynamic plasma by coupling with a radio-frequency electromagnetic field. Appl. Phys. Lett. 86, 191502. 10.1063/1.1926410 CrossRef
9. Nakamura, T. & Riedmuller, W. 1974 Stability of nonequilibrium MHD plasma in the regime of fully ionized seed. AIAA J. 12, 661 – 668. 10.2514/3.49316 CrossRef
10. NEA 2022 High-Temperature Gas-Cooled Reactors and Industrial Heat Applications. OECD Publishing.
11. Petit, J.P., Caressa, J.P. & Valensi, J. 1968 Theoretical and experimental study, using a shock tube, of the phenomena accompanying equilibrium breakdown in a closed cycle MHD generator. In Electricity from MHD: Proceedings of a Symposium on magnetohydrodynamic electric power generation (in French), vol. 2, p. 745.
12. Petit, J.P. & Geffray, J. 2008 Non-equilibrium plasma instabilities. In Proceedings of the, 2nd Euro-Asian Pulsed Power Conference, p. 1170.
13. Petit, J.P. & Valensi, J. 1969 Growth rate of electrothermal instability and critical Hall parameter in closed-cycle MHD generators when the electron mobility is variable. Compt. Rend. Acad. Sci. 269, 365.
14. Rosa, R.J. 1987 Magnetohydrodynamic Energy Conversion. Hemisphere Publishing Corporation.
15. Sheindlin, A.E., Batenin, V.A. & Asinovsky, E.I. 1964 Investigation of non-equilibrium ionization in a mixture of argon and potassium. In International Symposium on Magnetohydrodynamic Electric Power Generation, Paris, France.
16. Sorbom, B.M., et al. 2015 ARC: a compact, high-field, fusion nuclear science facility and demonstration power plant with demountable magnets. Fusion Eng. Des. 100, 378 – 405. 10.1016/j.fusengdes.2015.07.008 CrossRef
17. Sporn, P. & Kantrowitz, A. 1959 Large-scale generation of electric power by application of the magnetohydrodynamic concept. Power 103, 62.
18. Velikhov, E.P. 1962 Hall instability of current carrying slightly ionized plasmas. In 1st International Conference on MHD Electrical Power Generation, Newcastle upon Tyne, England, p. 135.
19. Velikhov, E.P. & Dykhne, A.M. 1963 Plasma turbulence due to the ionization instability in a strong magnetic field. In Proceedings of the, 6th International Conference on Phenomena in Ionized Gases, vol. 4 511, p. 511.
20. Velikhov, E.P., Dykhne, A.M. & Shipuk, I.Ya 1965 Ionization instability of a plasma with hot electrons. In: Proceedings of the, 7th International Conference on Phenomena in Ionized Gases.
21. Vieira, R.F., et al. 2024 Design, fabrication, and assembly of the SPARC toroidal field model coil. IEEE Trans. Appl. Supercon. 34, 0600615 –15. 10.1109/TASC.2024.3356571 CrossRef
22. Wesson, J. 2011 Tokamaks Fourth Edition. Oxford University Press.
