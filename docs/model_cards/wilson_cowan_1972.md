# Model card: Wilson & Cowan, 1972

Tags used below: **[paper p.N]** = read in the paper (page number of the journal);
**[computed]** = recomputed by us from the paper's equations, not stated in the paper;
**[interpretation]** = a choice we had to make because the paper is silent or ambiguous.

How the paper was read: the PDF is a 1972 scan with an OCR text layer. The OCR garbles
subscripts and Greek letters (e.g. "0." for θe, "a;" for ai, "9," for θi). Every equation
quoted here (3–16, 18–22, A1–A3) and every parameter value in the tables was checked against
the rendered page images, not only the OCR text.

---

## Reference

- Wilson, H. R. & Cowan, J. D. (1972). Excitatory and inhibitory interactions in localized
  populations of model neurons. *Biophysical Journal* 12, 1–24.
- DOI: 10.1016/S0006-3495(72)86068-5 (matches the PII in the PDF file name).
- Received 4 June 1971 [paper p.23]. Issue number is not printed in the PDF (publisher
  metadata says issue 1; check before citing).

---

## The paper contains several systems, not one

| Eq. | System | State | What it is | Use for the bench |
|---|---|---|---|---|
| 3–4 | Full integral model | E(t), I(t) | Delay equations: absolute refractory integral × sigmoid of a kernel-weighted input history. γ (sensitivity/excitation correlation) set to 0 [p.7]. | No. Not analysed further in the paper; kernel α(t) not specified. |
| 7–8 | Time coarse-grained model | Ē, Ī | Integrals replaced by r·Ē and k·Ē (eq. 6), Taylor expansion of E(t+τ). Has constants k, r, k′, r′, τ, τ′. | No (intermediate step). |
| 9–10 | Excitatory-only comparison | E | Used only for Fig. 3 (integral vs coarse-grained, r = 3 ms). No inhibition. | Could serve as a 1-D variant, but no full parameter set is given. |
| **11–12** | **Final coarse-grained E–I model** | E, I | The model analysed in all phase-plane, hysteresis and limit-cycle results. | **Yes. This is the model.** |
| A1–A3 | Extension with relative refractoriness | E (+ R(t,t′)) | Excitatory population only; threshold recovery θ(t−t′) left general. Reduces to eq. 7 when relative refractory period r = 0 [p.22]. | No (no parameters, no results). |

In addition, the response function can be any member of a sigmoid class [p.4]; eq. 15 (shifted
logistic) is the specific choice used for all figures. A multimodal (e.g. bimodal) response
function is discussed qualitatively [p.5, p.15–16]: it adds hysteresis loops, but no parameters
are given.

---

## Equations (copied, with paper numbering)

### Recommended system: eqs. 11, 12, 15 [p.10–11]

$$\tau_e \frac{dE}{dt} = -E + (k_e - r_e E)\,\mathcal{S}_e(c_1 E - c_2 I + P) \tag{11}$$

$$\tau_i \frac{dI}{dt} = -I + (k_i - r_i I)\,\mathcal{S}_i(c_3 E - c_4 I + Q) \tag{12}$$

$$\mathcal{S}(x) = \frac{1}{1+\exp[-a(x-\theta)]} - \frac{1}{1+\exp(a\theta)} \tag{15}$$

with separate (a_e, θ_e) and (a_i, θ_i). Maximum slope [p.11]:

$$\max[\mathcal{S}'(x)] = \mathcal{S}'(\theta) = \frac{a}{4} \tag{16}$$

Isoclines [p.10]:

$$c_2 I = c_1 E - \mathcal{S}_e^{-1}\!\left(\frac{E}{k_e - r_e E}\right) + P \quad (dE/dt = 0) \tag{13}$$

$$c_3 E = c_4 I + \mathcal{S}_i^{-1}\!\left(\frac{I}{k_i - r_i I}\right) - Q \quad (dI/dt = 0) \tag{14}$$

### Conditions stated in the paper (all derived with r_e = r_i = 1)

- Slope of the dE/dt = 0 isocline at the inflection point of S_e⁻¹: (c1/c2 − 9/(ae c2)) [p.11].
- (17) c1 > 9/ae — **sufficient** for the "kink" in the E-isocline, hence for ≥ 3 steady states for some constant (P, Q) (Theorem 1) [p.12].
- (18) ae c2 / (ae c1 − 9) > (ai c4 + 9) / (ai c3) — sufficient for 5 steady states for some (P, Q), not necessarily at the same time (Theorem 2) [p.14].
- (19) ae ai c2 c3 > (ae c1 − 9)(ai c4 + 9) — rewrite of 18 [p.14].
- (20) c1 ae > c4 ai + 18 — sufficient (not necessary) for instability of a single steady state [p.17].
- (21) = (18); (22) (ae c1 − 9)/(ae c2) < 1 — one steady state rather than five [p.17].
- Theorem 3: with 21 satisfied, if 20 fails → multiple hysteresis for some (P, Q); if 20 and 22 hold → limit cycle for some (P, Q) [p.18].
- Theorem 4: limit cycle for some (P, Q) ⇒ simple hysteresis for some other (P, Q) [p.20].
- c2 and c3 must be nonzero for non-trivial isoclines [p.10].

### Other systems (for completeness)

Eq. 3 and 4 [p.7]:

$$E(t+\tau) = \left[1 - \int_{t-r}^{t} E(t')\,dt'\right] \mathcal{S}_e\!\left\{\int_{-\infty}^{t} \alpha(t-t')\,[c_1E(t') - c_2I(t') + P(t')]\,dt'\right\} \tag{3}$$

$$I(t+\tau') = \left[1 - \int_{t-r'}^{t} I(t')\,dt'\right] \mathcal{S}_i\!\left\{\int_{-\infty}^{t} \alpha(t-t')\,[c_3E(t') - c_4I(t') + Q(t')]\,dt'\right\} \tag{4}$$

Eq. 7 and 8 [p.8]:

$$\tau \frac{d\bar E}{dt} = -\bar E + (1 - r\bar E)\,\mathcal{S}_e[k c_1 \bar E - c_2 k \bar I + k P(t)] \tag{7}$$

$$\tau' \frac{d\bar I}{dt} = -\bar I + (1 - r'\bar I)\,\mathcal{S}_i[k' c_3 \bar E - c_4 k' \bar I + k' Q(t)] \tag{8}$$

Eq. 9 and 10 (excitatory only) [p.8]:

$$\tau \frac{dE}{dt} = -E + \left[1 - \int_{t-r}^{t} E(t')\,dt'\right]\mathcal{S}_e\!\left\{\int_{-\infty}^{t} e^{-\alpha(t-t')}[c_1E(t') + P(t')]\,dt'\right\} \tag{9}$$

$$\tau \frac{d\bar E}{dt} = -\bar E + (1 - r\bar E)\,\mathcal{S}_e[k c_1 \bar E(t) + k P(t)] \tag{10}$$

Eq. A1–A3 [p.22]:

$$\frac{dR}{dt} = -\mathcal{S}[x(t), \theta(t-t')]\,R \tag{A1}$$

$$R(t,t') = E(t')\exp\left\{-\int_{t'}^{t}\mathcal{S}[x(t''), \theta(t''-t')]\,dt''\right\} \tag{A2}$$

$$\tau\frac{dE}{dt} = -E + \int_{t-r}^{t}\mathcal{S}[x(t),\theta(t-t')]R(t,t')\,dt' + \left[1 - r_eE - \int_{t-r}^{t}R(t,t')\,dt'\right]\mathcal{S}[x(t),\theta_0] \tag{A3}$$

---

## Nonlinearity

- General requirement [p.4]: S monotonically increasing on (−∞, ∞), asymptotes 0 and 1, exactly one inflection point (the "subpopulation threshold").
- Chosen form: shifted logistic, eq. 15, so that S(0) = 0 and E = I = 0 is a steady state when P = Q = 0 [p.9–11]. Because of the shift, S is slightly negative for x < 0 and its maximum is below 1.
- **k_e, k_i**: the paper defines them only in words as "the maximum values of the response functions" after the shift [p.10]. No formula or number is printed.
  **[interpretation]** k = S(+∞) = 1 − 1/(1 + exp(aθ)). For Fig. 4: k_e ≈ 0.966, k_i ≈ 0.982. Many later reproductions use k = 1; the difference matters (see below).
- The paper states that qualitative results (number/stability of steady states, hysteresis, limit cycles, Theorem 4) do not depend on the choice of logistic; the numerical conditions 17–22 do [p.11, p.21].

---

## Parameters

### Meaning and units

| Symbol | Meaning | Units | Source |
|---|---|---|---|
| E, I | Proportion of excitatory / inhibitory cells firing per unit time. 0 = low background activity; small negative values = depression below background. | dimensionless | [p.3] |
| c1, c2 | Average number of excitatory / inhibitory synapses per excitatory cell (E→E, I→E) | dimensionless | [p.5–6] |
| c3, c4 | Same for inhibitory cells (E→I, I→I) | dimensionless | [p.6] |
| P, Q | External input to E / I subpopulation; constant values define a "stimulus configuration" | same units as the sigmoid argument ("threshold units", Fig. 1 caption) | [p.6, p.12] |
| a_e, θ_e, a_i, θ_i | Slope parameter and position of maximum slope of S_e, S_i | 1/argument units; argument units | [p.11] |
| r_e, r_i | Absolute refractory term (coarse-grained) | — | Set to 1 "from now on" [p.12] |
| k_e, k_i | Maximum of S_e, S_i | dimensionless | [p.10], value see above |
| τ_e, τ_i | Time constants | msec | ≈ 10 ms for Fig. 10 [p.17]; "τ = 8 msec" in Fig. 9 and 11b captions; **not given for Figs. 4–8** |

Physiological values quoted in passing [p.9]: summation constant α ≈ 4 ms, absolute refractory period r ≈ 1–2 ms (Eccles 1964). These belong to the derivation, not to eqs. 11–12.

### Parameter sets given in the figure captions (all confirmed on the page images)

| Set | c1 | c2 | c3 | c4 | ae | θe | ai | θi | re, ri | τ | P, Q | Reported behaviour | Page |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Fig. 4 / 5** | 12 | 4 | 13 | 11 | 1.2 | 2.8 | 1 | 4 | 1, 1 | not given (Fig. 9 uses τ = 8 ms with these params) | Fig. 4: P = Q = 0. Fig. 5: P swept, Q = 0 | 3 steady states, 2 stable; simple hysteresis loop in P | 10, 13 |
| **Fig. 6** | 13 | 4 | 20 | 2 | 1.2 | 2.7 | 5 | 3.7 | 1, 1 | not given | P swept, Q = 0 | Two separate simple hysteresis loops, single stable state between them | 13 |
| **Fig. 7 / 8** | 13 | 4 | 22 | 2 | 1.5 | 2.5 | 6 | 4.3 | 1, 1 | not given | Fig. 7: P swept, Q = 0. Fig. 8: P = 0 | Two overlapping loops; three stable states around P = 0 (Fig. 8: 3 stable, 2 unstable) | 14 |
| Fig. 10 | 15 | 15 | 15 | 3 | 1 | 2 | 2 | 2.5 | not in caption | 10 ms | brief impulse | Damped oscillation of E − I | 16 |
| Fig. 11 / 12 | 16 | 12 | 15 | 3 | 1.3 | 4 | 2 | 3.7 | 1, 1 | 8 ms | Fig. 11: P = 1.25, Q = 0. Fig. 12: P swept ≈ 1–2 | Limit cycle; frequency and mean E increase with P | 18–19 |

Also: Fig. 1 illustrates an unshifted logistic with θ = 5, a = 1 [p.5]; Fig. 3 uses r = 3 ms for eqs. 9–10 with no other parameters given [p.10].

---

## Reported behaviour to reproduce

From the paper (qualitative; **no steady-state values are given in numbers**, only plots):

- Fig. 4: three intersections of the isoclines at P = Q = 0, labelled stable / unstable / stable.
- Fig. 5: hysteresis in E versus P (Q = 0). By eye: lower branch near E = 0, upper branch near E ≈ 0.45, loop roughly between P ≈ −0.35 and P ≈ 0.3 (rough reading of a 1972 plot).
- Fig. 6: two separated loops, the first around P ≈ −0.2…0.15, the second around P ≈ 0.4…0.9 (by eye).
- Fig. 7 / 8: three stable states around P = 0.
- Fig. 9: strength-duration curve for switching from the lower to the upper state (Fig. 5 parameters, τ = 8 ms).

### Recomputed steady states [computed]

Method: reduce eq. 12 = 0 to I(E) (unique root, the left side is monotone in I), scan E for roots
of eq. 11 = 0, classify with the Jacobian eigenvalues. τ_e = τ_i, r_e = r_i = 1, Q = 0,
k = S(+∞) unless stated. Script: `wc_check.py`.

| Set | P | Steady states (E, I) and stability |
|---|---|---|
| Fig. 4 | 0 | (0, 0) stable · (0.190, 0.068) saddle · (0.440, 0.226) stable |
| Fig. 6 | 0 | (0, 0) stable · (0.095, 0.000) saddle · (0.178, 0.116) stable focus |
| Fig. 7 / 8 | 0 | (0, 0) stable · (0.095, 0.000) saddle · (0.204, 0.189) stable focus · (0.380, 0.500) saddle · (0.454, 0.500) stable |
| Fig. 11 | 1.25 | single steady state (0.202, 0.107), unstable focus → limit cycle, consistent with Fig. 11 |

P intervals with ≥ 2 stable states (Q = 0, step 0.01):

| Set | k = S(+∞) | k = 1 |
|---|---|---|
| Fig. 4 / 5 | 2 stable on [−0.39, 0.30] | 2 stable on [−0.51, 0.28] |
| Fig. 6 | 2 stable on [−0.20, 0.15] and [0.42, 0.94] | [−0.24, 0.13] and [0.21, 0.87] |
| Fig. 7 / 8 | 2 on [−0.35, −0.14], **3** on [−0.13, 0.21], 2 on [0.22, 0.74] | 2 on [−0.36, −0.27], 3 on [−0.26, 0.20], 2 on [0.21, 0.71] |

These match the figures qualitatively for all three hysteresis sets. The k = S(+∞) version fits
the Fig. 5 and Fig. 6 loop edges better by eye; this is weak evidence, not proof.

Conditions checked against the sets [computed]:

| Set | (17) c1 > 9/ae | (18) | (20) | Consistent with paper? |
|---|---|---|---|---|
| Fig. 4 | 12 > 7.5 ✓ | 0.89 > 1.54 ✗ | — | Yes: 3 steady states, no 5 |
| Fig. 6 | 13 > 7.5 ✓ | 0.73 > 0.19 ✓ | 15.6 > 28 ✗ | Yes: multiple hysteresis (Theorem 3) |
| Fig. 7 | 13 > 6 ✓ | 0.57 > 0.16 ✓ | 19.5 > 30 ✗ | Yes: multiple hysteresis |
| Fig. 11 | 16 > 6.9 ✓ | 1.32 > 0.50 ✓; (22) 0.76 < 1 ✓ | 20.8 > 24 ✗ | Limit cycle exists although the sufficient condition 20 fails. Not a contradiction (20 is only sufficient), but the paper's own example does not satisfy it. |

### Jacobian used by the checker [computed from eqs. 11–12]

With x_e = c1E − c2I + P, x_i = c3E − c4I + Q, S′(x) = a σ(1 − σ), σ = 1/(1 + exp(−a(x − θ))):

- J11 = [−1 − r_e S_e(x_e) + (k_e − r_eE) c1 S_e′(x_e)] / τ_e
- J12 = −(k_e − r_eE) c2 S_e′(x_e) / τ_e
- J21 = (k_i − r_iI) c3 S_i′(x_i) / τ_i
- J22 = [−1 − r_i S_i(x_i) − (k_i − r_iI) c4 S_i′(x_i)] / τ_i

---

## Free parameters in our task, and ranges

The paper does not give fitting ranges or priors. Everything in this section is **our proposal**.

- Free (8): c1, c2, c3, c4, a_e, θ_e, a_i, θ_i. Optional 9th: P (the input at which bistability is required).
- Fixed: r_e = r_i = 1 [p.12], τ_e = τ_i (e.g. 10 ms, cf. p.17), Q = 0, k = S(+∞).
- Span of the paper's own five sets: c1 12–16, c2 4–15, c3 13–22, c4 2–11, a_e 1–1.5, θ_e 2–4, a_i 1–6, θ_i 2.5–4.3. A search box should be wider than this so the task is not trivial (e.g. c ∈ [0, 25], a ∈ [0.5, 7], θ ∈ [1, 6]); the box width is a difficulty knob.
- Hard constraints from the paper: c_j > 0, a > 0; c2 and c3 nonzero.
- Condition 17 is a cheap diagnostic (sufficient condition for the kink), not a valid filter: it is not necessary.

---

## Observed populations (what targets are given on)

- State variables: E and I, both dimensionless proportions. Hysteresis figures plot **E only**; I appears in the phase planes (Figs. 4, 8, 11a).
- Proposed target: the two stable values of E (low, high) at a fixed (P, Q), optionally with the matching I values.
- Two consequences of the equations that matter for target design [computed]:
  - At P = Q = 0, E = I = 0 is a steady state by construction (S(0) = 0). A "low state" target of 0 at P = Q = 0 is therefore free; the fit only has to create the high state. Use P ≠ 0 (or a nonzero low target) if the low state should be non-trivial.
  - With r_e = 1, E is bounded by k_e/r_e < 1, and at full saturation E → k_e²/(1 + k_e) ≈ 0.48 (similarly I → ≈ 0.5, visible in the Fig. 7/8 upper states where I = 0.500). High targets above ≈ 0.48 are infeasible.

---

## Ambiguities and interpretations

1. **k_e, k_i**: defined in words only. Interpretation k = S(+∞). Changes steady-state values by a few hundredths and moves loop edges by up to ≈ 0.1 in P (Fig. 4 set: left edge −0.39 vs −0.51).
2. **τ_e, τ_i not given for Figs. 4–8.** Fixed points do not depend on τ, but stability of foci does. Recomputed with τ_i/τ_e = 2 or 5: the intermediate stable focus of Fig. 6 and of Fig. 7/8 becomes **unstable**; with 0.5 or 1 it stays stable. The Fig. 4 states are robust (nodes and a saddle). Interpretation: τ_e = τ_i, consistent with the single τ quoted for Figs. 9 and 11b. This must be fixed in the config, not left free.
3. **Notation clashes in the paper**:
   - k (eq. 6, coarse-graining constant multiplying the input) versus k_e, k_i (eq. 11, maximum of S). Between eq. 7 and eq. 11 the coarse-graining k disappears from the sigmoid argument; it has presumably been absorbed into c1–c4, P, Q. Not stated.
   - r: absolute refractory period in the main text, duration of the **relative** refractory period in the Appendix.
   - α: kernel α(t) in eq. 2–4, a rate in the exponent of eq. 9, and "around 4 msec" (a time) on p.9. The dimensions do not agree as written.
4. **Fig. 10 parameter set**: with the caption values and r_e = r_i = 1 (not in the caption), our recomputation gives the rest state (0, 0) as a **saddle** at P = Q = 0, with two stable states at E ≈ −0.055 and E ≈ 0.045. This conflicts with the paper's requirement that the rest state be stable [p.9]. Either the caption, our reading of r_e, r_i, or the k interpretation differs from what was used. Irrelevant for bistability fitting; do not use this set.
5. **No numbers for steady states, loop edges or stability** are printed; all "reported behaviour" is qualitative or read from plots. Our recomputed values are ours.
6. **Stability method** is described only as "linearization … simple but tedious" [p.12]; the paper does not show it. The Jacobian above is our derivation.
7. **γ = 0** is an explicit modelling choice [p.7], not a derived result.
8. Fig. 7 caption says "dotted lines" for unstable states, Fig. 5 and 6 say "dashed". Cosmetic.
