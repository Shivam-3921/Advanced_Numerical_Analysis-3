---
tags: [comps, numerical-analysis, MTH653, DG, finite-volume, conservation-laws, H-div, H-curl]
aliases: [MTH 653 Study Guide]
---
# MTH 653 Advanced Numerical Analysis: Comps Study Guide
Spring 2025, Portland State University (W. Pazner). Sources: lecture notes 1-10 (PDF), `riemann.ipynb`, `godunov.ipynb`, and homework 1-4 (proofs and coding omitted; only results and ideas kept).
**Contents**
- [[#0. Notation and standard tools]]
- [[#1. Advection and the finite volume method (L1)]]
- [[#2. DG method for advection (L2)]]
- [[#3. Hyperbolic conservation laws (L3)]]
- [[#4. Finite volume for conservation laws and systems (L4)]]
- [[#5. DG for conservation laws and Godunov's theorem (L5)]]
- [[#6. DG for elliptic problems (L6)]]
- [[#7. Preconditioning DG (L7)]]
- [[#8. H(div) and H(curl) (L8)]]
- [[#9. Raviart-Thomas elements and Darcy (L9)]]
- [[#10. Nedelec elements and the de Rham complex (L10)]]
- [[#11. Quick reference: rates, key formulas, exam checklist]]
## 0. Notation and standard tools
### Mesh, spaces, jumps, averages
- Mesh $\mathcal T=\{\kappa\}$, skeleton $\Gamma=\Gamma_0\cup\Gamma_\partial$ (interior edges and boundary edges). Interior edge $e=\partial\kappa^-\cap\partial\kappa^+$ with traces $u^\pm$ and outward normals $n^\pm$ ($n^-=-n^+$).
- DG space: $V_h=\{v\in L^2(\Omega): v|_\kappa\in P_p(\kappa)\}$. Vector version $\Sigma_h=[V_h]^d$. System version $[V_h]^m$.
- Broken Sobolev space: $H^m(\mathcal T)=\{v\in L^2: v|_\kappa\in H^m(\kappa)\}$, broken derivatives $D_h^\alpha$, broken gradient $\nabla_h$, broken divergence $\nabla_h\cdot$.
- Average: $\{q\}=\tfrac12(q^-+q^+)$ (on $\partial\Omega$: $\{q\}=q$).
- Jump of a scalar (vector valued): $[\![q]\!]=q^-n^-+q^+n^+$ (on $\partial\Omega$: $qn$).
- Jump of a vector (scalar valued): $[\![\varphi]\!]=\varphi^-\cdot n^-+\varphi^+\cdot n^+$ (on $\partial\Omega$: $\varphi\cdot n$).
- Scalar "product rule" for jumps: $\{u\}[\![u]\!]=\tfrac12[\![u^2]\!]$.
- Element-by-element integration by parts identity:
  $$\sum_{\kappa}\int_{\partial\kappa} q\,\varphi\cdot n\,ds=\int_\Gamma\big([\![q]\!]\cdot\{\varphi\}+\{q\}[\![\varphi]\!]\big)ds$$
### Tools used over and over
- **Galerkin orthogonality** (consistency): $a(u-u_h,v_h)=0$ for all $v_h\in V_h$.
- **Cea's lemma / quasi-optimality** (continuity + coercivity): $|||u-u_h|||\lesssim\inf_{v_h}|||u-v_h|||$.
- **Aubin-Nitsche duality** for $L^2$ error: needs elliptic regularity $|\psi|_{H^2}\lesssim\|u-u_h\|_{L^2}$ and adjoint consistency.
- **Inverse inequality** (shape-regular mesh, $u_h\in V_h$): $\|\nabla u_h\|_{L^2(\kappa)}\lesssim h^{-1}\|u_h\|_{L^2(\kappa)}$.
- **Discrete trace inequality** (HW3 Q6, L7 Prop. 1): $\|u_h\|_{L^2(\partial\kappa)}\lesssim h^{-1/2}\|u_h\|_{L^2(\kappa)}$.
- **Continuous trace with scaling**: for $\theta\in H^1(\kappa)$, $\|\theta\|^2_{L^2(\partial\kappa)}\lesssim h^{-1}\|\theta\|^2_{L^2(\kappa)}+h\|\nabla\theta\|^2_{L^2(\kappa)}$.
- Proof template for both: scale to reference element $\kappa=h\hat\kappa$, use equivalence of norms on the finite-dimensional space $P_p(\hat\kappa)$, scale back (volume factor $h^d$, gradient factor $h^{-1}$, surface factor $h^{d-1}$).
- **Local $L^2$ projection** $\Pi_h:L^2\to V_h$ ($\int_\kappa(g-\Pi_hg)v_h=0$ on each $\kappa$). For $u\in H^{p+1}$:
  $$\|u-\Pi_hu\|_{H^m(\kappa)}\lesssim h^{p+1-m}|u|_{H^{p+1}(\kappa)},\quad \|u-\Pi_hu\|_{L^\infty(\kappa)}\lesssim h^{p+1}|u|_{W^{p+1}_\infty(\kappa)},\quad \|u-\Pi_hu\|_{L^2(\partial\kappa)}\lesssim h^{p+1/2}|u|_{H^{p+1}(\kappa)}$$
- **Young's inequality with weight**: $ab\le\tfrac12(\epsilon a^2+b^2/\epsilon)$.
## 1. Advection and the finite volume method (L1)
### Model problems
- Time-dependent advection with prescribed velocity $\beta:\Omega\to\mathbb R^d$:
  $$u_t+\nabla\cdot(\beta u)=0,\quad u(x,0)=u_0,\quad u=g\text{ on }\Gamma_{in}$$
- Inflow boundary: $\Gamma_{in}=\{x\in\partial\Omega:\beta\cdot n<0\}$ ($n$ outward normal).
- Steady advection-reaction:
  $$\nabla\cdot(\beta u)+cu=f\ \text{in }\Omega,\quad u=g\ \text{on }\Gamma_{in},\qquad c\in L^\infty,\quad c+\tfrac12\nabla\cdot\beta\ge\gamma_0>0$$
- The equation is **hyperbolic** (unlike elliptic Poisson/elasticity/Stokes or parabolic heat). Information travels along characteristics (direction $\beta$).
- Time-dependent and steady problems are interchangeable: backward Euler on $u_t+\nabla\cdot(\beta u)=0$ gives $\Delta t\,\nabla\cdot(\beta u^{n+1})+u^{n+1}=u^n$, which has the form of the steady problem with $c=1$. In practice explicit time stepping is more common for the time-dependent problem.
### $H^1$-conforming discretization
- Strong BC: $V=\{v\in H^1:v=0\text{ on }\Gamma_{in}\}$ (for $g=0$). Find $u\in V$: $\int\nabla\cdot(\beta u)v+\int cuv=\int fv$ for all $v\in V$.
- Weak BC: $u,v\in H^1(\Omega)$ and
  $$a(u,v)=\int_\Omega\nabla\cdot(\beta u)v\,dx-\int_{\Gamma_{in}}uv\,\beta\cdot n\,ds+\int_\Omega cuv\,dx,\qquad F(v)=-\int_{\Gamma_{in}}gv\,\beta\cdot n\,ds+\int_\Omega fv\,dx$$
- Non-symmetric form. Key identity: $\nabla\cdot(\beta u)u=\tfrac12(\nabla\cdot\beta)u^2+\tfrac12\nabla\cdot(\beta u^2)$.
- **Lemma (coercivity in a weak norm):**
  $$a(u,u)\ge\gamma_0\|u\|^2_{L^2}+\tfrac12\int_{\partial\Omega}|\beta\cdot n|u^2ds$$
- Coercive w.r.t. $|||u|||^2=\|u\|_{L^2}^2+\int_{\partial\Omega}|\beta\cdot n|u^2$, which is **not** a norm on $H^1$ (no gradient control), so the usual FE theory is only partly available.
- Galerkin orthogonality: $a(u-u_h,v_h)=0$.
- **Theorem:** if $u\in H^2$ and $V_h$ = continuous piecewise linears,
  $$\|u-u_h\|_{L^2}^2+\int_{\partial\Omega}|\beta\cdot n||u-u_h|^2\lesssim h^2|u|_{H^2}^2$$
- So the $L^2$ error is $O(h)$: **suboptimal by one order** (interpolation gives $h^2$). If $u\in H^1\setminus H^2$ the method does not converge in $H^1$.
- **Numerical symptom: spurious oscillations.** $\beta=1$, $u_0=e^{-150(x-1/2)^2}$, periodic, RK4 to $T=3$: the shape is roughly right but there are wiggles. Motivates FV, then DG.
### Finite volume (FV) method
- Unknowns: cell averages $u_i=\frac1{|\kappa_i|}\int_{\kappa_i}u\,dx$, $h_i=|\kappa_i|$ (compare: FE stores vertex values).
- Integrate the PDE over a cell and use the divergence theorem:
  $$h_i\frac{du_i}{dt}+\int_{\partial\kappa_i}u\,\beta\cdot n\,ds=0\qquad(n\text{ outward normal of }\kappa_i)$$
- **Key approximation:** only cell averages are known, so the boundary trace of $u$ is replaced by the **upwind value**: the cell average of the neighbor on the side the characteristics come from. For an edge $e$ of $\kappa_i$ shared with neighbor $\kappa_j$ ($n$ = outward normal of $\kappa_i$): $u_e=u_i$ if $\beta\cdot n\ge0$ (outflow edge) and $u_e=u_j$ if $\beta\cdot n<0$ (inflow edge). Result: a coupled ODE system
  $$h_i\frac{du_i}{dt}+\sum_{e\subset\partial\kappa_i}|e|\,u_e\,\beta\cdot n=0$$
- 1D, $\beta\equiv1$, $\kappa_i=[x_i,x_{i+1}]$: 
  $$h_i\frac{du_i}{dt}+u_i-u_{i-1}=0\iff u_i'=-\frac{u_i-u_{i-1}}{h_i}$$
which is the **backward difference** for $u_x$. First-order accurate (Taylor).
- **Directionality matters:** for $\beta=1$ use backward difference; forward or centered differences are unstable. For $\beta=-1$ use forward difference.
### Homework 1: stability of difference schemes for advection
- Fully discrete (forward Euler in time, $\beta=1$, periodic, $\Delta t=h/2$):
  $$u_i^{n+1}=u_i^n-\Delta t\frac{u_i^n-u_{i-1}^n}{h_i},\qquad \boldsymbol u^{n+1}=(I-\Delta tA)\boldsymbol u^n$$
with $A=\frac1h(I-S)$, $S$ the periodic shift (wrap-around indices).
- Observed/expected: upwind (backward) FV with forward Euler is stable and **first-order** (error halves when $n$ doubles, so $L^2$ error ratio $\approx2$). Forward difference and centered difference blow up.
- **Forward Euler stability for $u'=\lambda u$, $\mathrm{Re}\lambda<0$:** the discrete solution decays iff
  $$|1+\lambda\Delta t|\le1$$
i.e. $\lambda\Delta t$ lies in the disk of radius 1 centered at $-1$ in $\mathbb C$.
- Spectra of $-\Delta tA$ (circulant matrices, Fourier modes $e^{ij\theta}$, $\nu=\Delta t/h$):
  - Backward (upwind): $-\nu(1-e^{-i\theta})$, circle of radius $\nu$ centered at $-\nu$. Inside the stability disk iff $\nu\le1$ (CFL condition). Stable.
  - Forward (downwind): $\nu(1-e^{i\theta})$, circle centered at $+\nu$ in the right half-plane. Outside the disk, unstable.
  - Centered: $-i\nu\sin\theta$, purely imaginary, $|1+iy|>1$. Unstable (growth like $\exp(c\,T\nu^2/\Delta t)$ as $h\to0$ with $\nu$ fixed).
- Moral: explicit time stepping + spatial operator requires the spectrum of $-\Delta t A$ to lie in the stability region. Upwind shifts the spectrum into the left half-plane (dissipation).
### Homework 1 Q1: consistency of DG (see also [[#2. DG method for advection (L2)]])
- Result: the DG variational formulation is consistent iff the numerical flux $\widehat F$ is consistent ($\widehat F(u,u,n)=u\beta$). Proof is by integrating by parts back element-wise (periodic BC, $\Gamma_\partial=\varnothing$).
## 2. DG method for advection (L2)
### Derivation of the formulation
- Test with $v\in H^1(\mathcal T)$, integrate over each $\kappa$, integrate by parts:
  $$-\int_\kappa u\beta\cdot\nabla v\,dx+\int_{\partial\kappa}uv\,\beta\cdot n\,ds+\int_\kappa cuv\,dx=\int_\kappa fv\,dx$$
- $u$ is double-valued on edges. **Replace $u^-\beta$ by a numerical flux** $\widehat F(u^-,u^+,n^-)$ with two properties:
  - **Consistent:** $\widehat F(u,u,n)=u\beta$ for all $u,n$.
  - **Conservative:** $\widehat F(u^-,u^+,n^-)=\widehat F(u^+,u^-,n^+)$ (single-valued flux, same on both sides).
- Summing over elements (each interior edge appears twice, combine using conservativity and the jump $[\![v]\!]=n^-v^-+n^+v^+$): find $u\in H^1(\mathcal T)$ with
  $$a(u,v)=-\int_\Omega u\beta\cdot\nabla_hv\,dx+\int_\Gamma\widehat F\cdot[\![v]\!]\,ds+\int_\Omega cuv\,dx=\int_\Omega fv\,dx=F(v)$$
- On $\Gamma_\partial$ use $\widehat F_\partial(u,n)$ (weak enforcement of BC as in the conforming case). Notation $\widehat F$ means either on interior or boundary edges.
- **Proposition 1:** the formulation is consistent (exact solution $u$ of the PDE satisfies $a(u,v)=F(v)$ for all $v\in H^1(\mathcal T)$) iff $\widehat F$ is consistent.
- DG discretization: find $u_h\in V_h$ with $a(u_h,v_h)=F(v_h)$ for all $v_h\in V_h$. Non-conforming: $V_h\subset H^1(\mathcal T)$ but $V_h\not\subset H^1(\Omega)$.
### Numerical fluxes for advection
- Mean-value (central) flux: $\widehat F_{mv}=\{u\}\beta$. Natural-looking but **bad** for DG.
- Upwind flux: $\widehat F_{uw}=u^-\beta$ if $\beta\cdot n^-\ge0$, else $u^+\beta$.
- **General flux** with parameter $b_0$:
  $$\widehat F_{b_0}(u^-,u^+,n^-)=\big(\{u\}+b_0\cdot[\![u]\!]\big)\beta,\qquad b_0=b_0\,n^-\,\mathrm{sgn}(\beta\cdot n^-),\ \ 0\le b_0\le\tfrac12$$
- $b_0=0$ gives mean-value; $b_0=\tfrac12$ gives upwind. Useful identity: $(b_0\cdot n^-)(\beta\cdot n^-)=b_0|\beta\cdot n^-|$.
### Coercivity and stability (periodic BC)
- **Theorem 1.** With $\widehat F=\widehat F_{b_0}$ and $c_0^2:=c+\tfrac12\nabla\cdot\beta\ge\gamma_0$,
  $$a(u,u)=\|c_0u\|^2_{L^2(\Omega)}+\int_\Gamma b_0|\beta\cdot n^-|\,[\![u]\!]^2ds=:|||u|||^2_{b_0}$$
- Ingredients: $-\int_\kappa u\beta\cdot\nabla u=-\frac12\int_{\partial\kappa}\beta\cdot nu^2+\frac12\int_\kappa u^2\nabla\cdot\beta$, and $\{u\}[\![u]\!]=\tfrac12[\![u^2]\!]$ cancels the mean-value part.
- DG norm: $|||u|||^2_{b_0}=\|c_0u\|^2_{L^2}+\|(b_0\cdot\beta)^{1/2}[\![u]\!]\|^2_{L^2(\Gamma)}=a(u,u)$.
- **Stability:** $|||u|||_{b_0}\lesssim\|f\|_{L^2}$ (since $|||u|||^2=F(u)\le\|f\|\|u\|$). For $b_0=0$ this is just $L^2$ stability. For $b_0\ne0$ it is stability in a **stronger norm** (includes jumps).
- **Well-posedness of the discrete problem:** coercivity implies $a(u_h,u_h)=0\Rightarrow u_h=0$, so the linear system is nonsingular (unique discrete solution).
### Error estimates
- **Theorem 2.** Let $u\in H^{p+1}(\Omega)$ be the exact solution, $u_h$ the DG solution.
  - If $b_0$ is bounded uniformly from below (e.g. upwind): 
    $$|||u-u_h|||_{b_0}\lesssim h^{p+1/2}|u|_{H^{p+1}(\Omega)}$$
  - If $b_0=0$ (mean-value flux):
    $$|||u-u_h|||_{b_0}\lesssim h^{p}|u|_{H^{p+1}(\Omega)}$$
- **Proof idea:** split $e=(u-\tilde u)-(u_h-\tilde u)$, $\tilde u=\Pi_hu$. Coercivity + Galerkin orthogonality give $|||u_h-\tilde u|||^2=a(\eta,\xi)$ with $\eta=u-\tilde u$, $\xi=u_h-\tilde u$.
  - Reaction term: $\lesssim h^{p+1}|u|\,|||\xi|||$.
  - Advection volume term: use degree-0 projection $\Pi_0\beta$ (so $\int\eta\,\Pi_0\beta\cdot\nabla_h\xi=0$), then $\|\beta-\Pi_0\beta\|_\infty\lesssim h$ and an inverse inequality.
  - Face terms: with $b_0$ bounded below, the jump term $[\![\xi]\!]$ is controlled by the DG norm, giving $h^{p+1/2}$. With $b_0=0$ no control of jumps, so use the inverse/trace inequality and lose half an order ($h^p$).
- Takeaway: **upwinding gives extra jump control and an extra $h^{1/2}$**; central flux is stable in $L^2$ only and has suboptimal rate.
## 3. Hyperbolic conservation laws (L3)
### Setup
- General form (Cauchy problem):
  $$u_t+\nabla\cdot F(u)=0,\quad u(x,0)=u_0(x)$$
with $u\in\mathbb R^m$ and flux $F:\mathbb R^m\to\mathbb R^{m\times d}$. Linear advection: $m=1$, $F(u)=\beta u$.
- **Conservation property:** for $D\subseteq\Omega$,
  $$\frac{d}{dt}\int_Du\,dx=-\int_{\partial D}F(u)\cdot n\,ds$$
(change of the conserved quantity equals the boundary flux).
### Method of characteristics (scalar, 1D)
- $u_t+F(u)_x=0$. Characteristic ODE $\dot\chi_s(t)=F'(u(\chi_s(t),t))$, $\chi_s(0)=s$.
- $u$ is **constant along characteristics**: $u(\chi_s(t),t)=u_0(s)$. Hence characteristics are straight lines
  $$\chi_s(t)=s+F'(u_0(s))\,t$$
- Linear advection: parallel lines of slope $\beta$.
- **Burgers' equation:** $F(u)=\tfrac12u^2$, $u_t+\tfrac12(u^2)_x=0$, $F'(u)=u$. Characteristic from $s$ has slope $u_0(s)$ (speed).
  - $u_0$ increasing: characteristics don't cross, classical solution for all time.
  - $u_0$ decreasing somewhere: characteristics cross, solution becomes multivalued. **Shocks form in finite time even for smooth data.**
- Breaking time (Burgers, from $\partial_s\chi_s(t)=1+u_0'(s)t=0$): 
  $$t^*=-\frac1{\min_su_0'(s)}$$
(and $t^*=\infty$ if $u_0'\ge0$).
### Homework 2: characteristics for Burgers
- Ramp up: $u_0=0\ (x\le0)$, $x\ (0\le x\le1)$, $1\ (x\ge1)$. Characteristics $x=s(1+t)$ in the ramp, so
  $$u(x,t)=\begin{cases}0&x\le0\\ \dfrac{x}{1+t}&0\le x\le1+t\\ 1&x\ge1+t\end{cases}$$
Valid for all $T$ (no crossing, $T=\infty$): rarefaction-like smoothing.
- Ramp down: $u_0=1\ (x\le0)$, $1-x\ (0\le x\le1)$, $0\ (x\ge1)$. Characteristics $x=s+(1-s)t$, so
  $$u(x,t)=\begin{cases}1&x\le t\\ \dfrac{1-x}{1-t}&t\le x\le1\\ 0&x\ge1\end{cases}$$
Valid for $t<1$; characteristics all meet at $(x,t)=(1,1)$. Maximal $T=1$ (matches $t^*=-1/\min u_0'=1$).
### Weak solutions
- Multiply by test function $\varphi\in C^1_0$ and integrate by parts. $u$ is a **weak solution** if for all $\varphi$ with compact support in $D\times[0,T)$:
  $$\int_0^T\!\!\int_D\big(u\,\varphi_t+F(u)\cdot\nabla\varphi\big)dx\,dt+\int_D\varphi(x,0)u_0(x)\,dx=0$$
- HW2 Q3: a **smooth weak solution is a classical solution**: integrating back by parts with test functions supported away from $t=0$ gives $u_t+F(u)_x=0$; test functions not vanishing at $t=0$ then give $u(0,x)=u_0(x)$.
- **Weak solutions are not unique.** Burgers with $u_0=H(x)=\mathbb 1_{x\ge0}$ has two weak solutions:
  $$u_1=H(x-t/2)\ \ (\text{shock}),\qquad u_2=\begin{cases}0&x<0\\ x/t&0<x<t\\ 1&x>t\end{cases}\ \ (\text{rarefaction})$$
### Rankine-Hugoniot (HW2 Q4-Q5)
- Suppose a smooth curve $C$ splits the domain into $V_L,V_R$ and $u$ is smooth on each side with traces $u_L,u_R$ (and fluxes $F_L,F_R$) on $C$. Plugging into the weak form gives a boundary integral over $C$ of $[(u_L-u_R)n_t+(F_L-F_R)n_x]\varphi\,ds$, which must vanish for all $\varphi$.
- With $C=\{(t,x(t))\}$, $x'=\sigma$ (the **shock speed**), $(n_t,n_x)\propto(-\sigma,1)$:
  $$\boxed{F_L-F_R=\sigma(u_L-u_R)}\qquad\Longleftrightarrow\qquad\sigma=\frac{F(u_L)-F(u_R)}{u_L-u_R}$$
- Burgers: $\sigma=\tfrac12(u_L+u_R)$.
- Same formula derived by conservation: $\frac{d}{dt}\int_{-a}^au\,dx=F(u(-a))-F(u(a))$ and $\int u=(a+\sigma t)u_L+(a-\sigma t)u_R$.
### Entropy condition and the physically relevant solution
- **Entropy condition** (from: going backward along a characteristic you should never hit a discontinuity), a shock must be **compressive**:
  $$F'(u_L)>\sigma>F'(u_R)$$
Burgers: $u_L>\sigma>u_R$.
- Among the many weak solutions there is a unique **entropy / physically relevant** one: the **vanishing viscosity limit**. Let $u^\epsilon$ solve $u^\epsilon_t+\nabla\cdot F(u^\epsilon)-\epsilon\Delta u^\epsilon=0$ (unique for each $\epsilon>0$). Then
  $$\lim_{\epsilon\to0}\|u-u^\epsilon\|_{L^1(\Omega\times[0,T])}=0$$
- Numerical goal: methods that converge to this solution.
- HW2 Q6 ($u_0=H$ for Burgers): both $u_1$ (shock, $\sigma=\tfrac12$) and $u_2$ (fan) satisfy Rankine-Hugoniot ($F_L-F_R=0-\tfrac12=-\tfrac12=\sigma(u_L-u_R)=\tfrac12\cdot(-1)$). Entropy check for the shock: $u_L>\sigma>u_R$ means $0>\tfrac12>1$, **false**. The shock violates entropy (an expansion shock); the rarefaction is continuous and is the **entropy solution**.
### Riemann problem (scalar, convex flux)
- Data: $u(x,0)=u_L$ for $x\le0$, $u_R$ for $x>0$, $u_L\ne u_R$.
- **Self-similarity:** $u(x,t)=w(x/t)=w(\xi)$ because $u_\lambda(x,t)=u(\lambda x,\lambda t)$ solves the same problem. Substituting gives (for $w'\ne0$)
  $$F'(w(\xi))=\xi$$
- Assume $F$ convex ($F'$ increasing).
- $u_L>u_R$: **shock** (entropy admissible)
  $$u(x,t)=\begin{cases}u_L&x/t\le\sigma\\ u_R&x/t>\sigma\end{cases},\quad \sigma=\frac{F(u_L)-F(u_R)}{u_L-u_R}$$
- $u_L<u_R$: **rarefaction wave** (expansion fan)
  $$u(x,t)=\begin{cases}u_L&x/t\le F'(u_L)\\ (F')^{-1}(x/t)&F'(u_L)<x/t<F'(u_R)\\ u_R&x/t\ge F'(u_R)\end{cases}$$
- `riemann.ipynb` implements exactly this with $F(u)=u^2/2$, $F'(u)=u$, $(F')^{-1}(u)=u$, $u_L=-0.5$, $u_R=1$ (fan between speeds $-0.5$ and $1$), and plots the wave curves $x=\sigma t$ or $x=F'(u_{L,R})t$ in the $(x,t)$ plane.
- For scalar laws the **maximum principle** holds: $u_0\in[a,b]\Rightarrow u(x,t)\in[a,b]$.
### Homework 2: conceptual results from the coding problems
- Godunov for Burgers with $u_0=e^{-10(x-1/8)^2}-e^{-10(x+1/8)^2}$ on $[-1,1]$, 200 cells, $\Delta t=h$, $T=1$. Qualitatively: the positive bump (right) and negative bump (left) steepen on their outer sides (decreasing profile) into shocks that travel outward, while the region between them (increasing profile) forms a rarefaction; the shocks then decay in amplitude.
- Piecewise constant data $u_0=-\tfrac12$ on $(-\tfrac12,0)$, $\tfrac12$ on $(0,\tfrac12)$, $0$ elsewhere has **three waves**:
  - At $x=-\tfrac12$: $u_L=0>u_R=-\tfrac12$, shock with $\sigma=-\tfrac14$.
  - At $x=0$: $u_L=-\tfrac12<u_R=\tfrac12$, rarefaction fan spreading with speeds from $-\tfrac12$ to $\tfrac12$.
  - At $x=\tfrac12$: $u_L=\tfrac12>u_R=0$, shock with $\sigma=\tfrac14$.
- The shocks reach the fan edges at $t=2$, so for $T\le1$ the three waves do not interact.
- Local Lax-Friedrichs (LLF) vs Godunov: LLF is more diffusive (smears shocks), but needs no Riemann solver, only a bound on the max wave speed (cheap and simple). See [[#4. Finite volume for conservation laws and systems (L4)]].
## 4. Finite volume for conservation laws and systems (L4)
### FV scheme in 1D
- Cell averages $u_i(t)$, interface $x_{i+1/2}$. Integrating $u_t+F(u)_x=0$:
  $$h_i\frac{du_i}{dt}+F(u(x_{i+1}))-F(u(x_i))=0$$
Need to approximate the interface flux from cell averages: $\widehat F(u_i,u_{i+1})\approx F(u(x_{i+1/2}))$, giving
$$h_i\,u_i'+\widehat F(u_i,u_{i+1})-\widehat F(u_{i-1},u_i)=0$$
- For advection: upwinding ($\beta\ge0$: left cell value, else right).
### Godunov method
- **Godunov flux:** let $w$ be the self-similar entropy solution of the Riemann problem with data $(u_L,u_R)=(u_{i-1},u_i)$ at $x_i$; $w^*=w(x_i,t)$ for $t>0$ (well defined by self-similarity). Then
  $$\widehat F(u_L,u_R):=F(w^*(u_L,u_R))$$
- This uses an **exact Riemann solver**. For linear advection it reduces to the upwind flux (so Godunov = first-order upwind).
- Exact solver can be costly (even scalar needs $(F')^{-1}$, maybe Newton). Motivates **approximate Riemann solvers**.
- **Burgers:** $F=u^2/2$, $(F')^{-1}=\mathrm{id}$, $\sigma=\tfrac12(u_L+u_R)$. Value at $x/t=0$:
  $$w^*=\begin{cases}u_L&u_L\ge u_R,\ u_L+u_R\ge0\\ u_R&u_L\ge u_R,\ u_L+u_R<0\\ u_L&u_L<u_R,\ u_L\ge0\\ 0&u_L<0<u_R\\ u_R&u_L<u_R,\ u_R\le0\end{cases}$$
- Compact form for convex $F$ with minimizer $u^*$ of $F$ (sonic point; $u^*=0$ for Burgers): $\widehat F_{God}(u_L,u_R)=\max\{F(\max(u_L,u^*)),F(\min(u_R,u^*))\}$. Equivalent: $\min_{[u_L,u_R]}F$ if $u_L\le u_R$, $\max_{[u_R,u_L]}F$ if $u_L>u_R$.
- Transonic rarefaction ($u_L<0<u_R$): the interface sees the sonic state $0$ and the Godunov flux is $F(0)=0$.
### Hyperbolic systems
- **Linear system:** $u_t+(Au)_x=0$, $u\in\mathbb R^m$, $F(u)=Au$. **Hyperbolic** iff $A$ is diagonalizable with real eigenvalues: $AR=R\Lambda$.
- Characteristic variables $w=R^{-1}u$ decouple the system into $m$ **scalar advection equations**
  $$w_t+\Lambda w_x=0$$
with speeds $\lambda_i$ (i.e., $m$ independent waves).
- Riemann solution for linear systems: expand $u_L=\sum_kw_k^Lr_k$, $u_R=\sum_kw_k^Rr_k$. The state at $x/t=0$ is (upwinding each characteristic variable)
  $$u^*=\sum_{\lambda_k>0}w_k^Lr_k+\sum_{\lambda_k<0}w_k^Rr_k,\qquad \widehat F_{God}=Au^*$$
- **Quasilinear form:** $u_t+A(u,x,t)u_x=0$ is hyperbolic at a point iff $A(u,x,t)$ is diagonalizable with real eigenvalues. A nonlinear conservation law $u_t+F(u)_x=0$ is the quasilinear system with $A=DF=(\partial F_i/\partial u_j)$ (Jacobian); it is hyperbolic iff $DF$ is.
### Example: isentropic gas dynamics
- Conservation of mass $\rho_t+(\rho u)_x=0$ and momentum $(\rho u)_t+(\rho u^2+p)_x=0$. Close with isentropic law $p=P(\rho)=\hat\kappa\rho^\gamma$, $\hat\kappa>0$, $\gamma>1$. Pressure is **not** a conserved variable.
- With $q=(q_1,q_2)=(\rho,\rho u)$:
  $$F(q)=\begin{pmatrix}q_2\\ q_2^2/q_1+P(q_1)\end{pmatrix},\qquad DF=\begin{pmatrix}0&1\\ -u^2+P'(\rho)&2u\end{pmatrix}$$
- Eigenvalues $\lambda_{1,2}=u\mp\sqrt{P'(\rho)}$, real iff $P'(\rho)\ge0$. Since $P'(\rho)=\hat\kappa\gamma\rho^{\gamma-1}>0$ for $\rho>0$, the system is hyperbolic for all physical states.
### Example: linear acoustics (linearization about $\rho_0,u_0$)
$$u_t+Au_x=0,\qquad A=\begin{pmatrix}u_0&K_0\\ 1/\rho_0&u_0\end{pmatrix},\quad K_0=\rho_0P'(\rho_0)$$
- Eigenvalues $u_0\pm\sqrt{K_0/\rho_0}$: one left-moving and one right-moving wave.
- Course example $u_0=0$, $\rho_0=\tfrac14$, $K=2$: $A=\begin{pmatrix}0&2\\4&0\end{pmatrix}$, $\lambda=\pm2\sqrt2$, eigenvectors $(\pm\tfrac1{\sqrt2},1)$.
### Godunov notebook: linear acoustics with Godunov's method
- Variables $q=(p,u)$ (pressure, velocity). Flux $F(q)=(Ku,\ p/\rho)$, $A=\begin{pmatrix}0&K\\1/\rho&0\end{pmatrix}$, wave speed $c=\sqrt{K/\rho}$, eigenvectors $r_{1,2}=(\pm\sqrt{K\rho},\,1)$ for $\lambda=\pm c$. $K=2$, $\rho=\tfrac14$. Initial: $p_0=e^{-100(x-0.35)^2}$, $u_0=0$ on $[0,1]$, $n=100$ cells.
- Riemann solver: $w_L=R^{-1}q_L$, $w_R=R^{-1}q_R$, $q^*=w_L[0]r_1+w_R[1]r_2$, flux $F(q^*)$.
- Update (FV, forward Euler):
  $$q_i^{n+1}=q_i^n-\frac{\Delta t}{h}\big(\widehat F_{i+1/2}-\widehat F_{i-1/2}\big)$$
- **CFL/time step restriction:** $\Delta t=h/c$ (with $c$ the max wave speed).
- Boundary: reflecting wall (ghost states chosen so the interface state has $u^*=0$).
- Plotting the characteristic variables $w=R^{-1}q$ shows the initial pressure bump splits into left- and right-moving halves that reflect at the walls.
### Local Lax-Friedrichs (LLF) flux
$$\widehat F(u_L,u_R)=\tfrac12\big(F(u_L)+F(u_R)\big)+\tfrac\alpha2(u_L-u_R)$$
- $\alpha$ is a bound on the max wave speed of the Riemann problem (linear system: $\max|\lambda_i|$; Burgers: $\max\{|u_L|,|u_R|\}$). No Riemann solve needed.
- In first-order FV, LLF is much more dissipative than Godunov (smears the solution): trades accuracy for simplicity. In DG the effect is greatly reduced, so LLF is an acceptable DG flux.
## 5. DG for conservation laws and Godunov's theorem (L5)
### Approximate (two-wave) Riemann solver
- Assume two waves of speeds $\sigma_1<\sigma_2$ separating $u_L$, $w^*$, $u_R$. Rankine-Hugoniot on each wave:
  $$F_L-F^*=\sigma_1(u_L-w^*),\qquad F^*-F_R=\sigma_2(w^*-u_R)$$
- Solving:
  $$w^*=\frac{F_L-F_R-\sigma_1u_L+\sigma_2u_R}{\sigma_2-\sigma_1},\qquad F^*=\frac{\sigma_1F(u_R)-\sigma_2F(u_L)-\sigma_1\sigma_2(u_R-u_L)}{\sigma_1-\sigma_2}$$
- $F^*\ne F(w^*)$ in general. Use $F^*$ as the numerical flux. Stability requires $\sigma_1\le c_{\min}$, $\sigma_2\ge c_{\max}$ (bounds on the Riemann wave speeds; linear system: $\lambda_{\min},\lambda_{\max}$).
### LLF as a special case
- $\sigma_1=-\alpha$, $\sigma_2=\alpha$, $\alpha\ge|c_i|$ for all wave speeds gives
  $$\widehat F(u^-,u^+)=\tfrac12\big(F(u_L)+F(u_R)\big)+\tfrac\alpha2(u_L-u_R)$$
- **LLF = central differencing + numerical diffusion.** For constant $h,\alpha$:
  $$\frac{du_i}{dt}=-\frac{F(u_{i+1})-F(u_{i-1})}{2h}+\frac{h\alpha}{2}\,\frac{u_{i+1}-2u_i+u_{i-1}}{h^2}$$
It approximates $u_t+F(u)_x=\epsilon u_{xx}$ with $\epsilon=h\alpha/2$: diffusion proportional to max wave speed, vanishing as $h\to0$.
### DG for systems
- Space $[V_h]^m$ (piecewise polynomials of degree $p$, no continuity). Take $u_h\in[V_h]^m$, test with $v_h$, integrate flux term by parts on each element:
  $$\sum_\kappa\int_\kappa\partial_tu_h\cdot v_h-\int_\kappa F(u_h):\nabla v_h+\int_{\partial\kappa}F(u_h)n\cdot v_h\,ds=0$$
- Replace $F(u_h)n$ on $\Gamma$ by a single-valued numerical flux $\widehat F(u^-,u^+,n)$:
  $$\int_\Omega\partial_tu_h\cdot v_h\,dx-\int_\Omega F(u_h):\nabla_hv_h\,dx+\sum_\kappa\int_{\partial\kappa}\widehat F(u^-,u^+,n)\cdot v_h^-\,ds=0\quad\forall v_h$$
- **Choosing $\widehat F$:** on edge $e$ with normal $n$, consider the 1D Riemann problem with 1D flux $G(w)=F(w)n$ and states $u_L=u^-,u_R=u^+$. Any 1D FV numerical flux $\widehat G$ can be used: $\widehat F=\widehat G\,$ (as a normal flux).
  - Godunov flux: $\widehat F=F(w^*)n$ with $w^*$ the exact Riemann solution.
  - Local Lax-Friedrichs: $\widehat F(u^-,u^+,n)=\tfrac12\big(F(u^-)+F(u^+)\big)n+\tfrac\alpha2(u^--u^+)$, $\alpha$ = max wave speed of the 1D problem.
  - Roe flux: $\widehat F=F(\tilde w)n$ with $\tilde w$ the solution of a linearized Riemann problem.
- The numerical flux matters **much less in DG (especially high $p$) than in FV**: for smooth solutions $[\![u_h]\!]=O(h^{p+1/2})$, and for continuous $u$ all consistent fluxes agree.
- **$p=0$ DG = first-order FV.** DG generalizes FV (higher order) and FE (suited to hyperbolic laws).
### Monotonicity, maximum principle, Godunov's theorem
- Scalar conservation laws satisfy the maximum principle. A method is **monotonicity preserving** if it creates no new extrema.
- **Godunov's theorem:** linear numerical methods that do not generate new extrema (monotonicity preserving) are at most **first-order accurate**.
- Definition (HW3): $\widehat F(u_L,u_R)$ is **monotone** if non-decreasing in $u_L$ and non-increasing in $u_R$, i.e. $\widehat F(\uparrow,\downarrow)$.
- First-order FV is monotonicity preserving if $\widehat F$ is monotone (true for Godunov, LLF).
- For $p>0$, DG applied to a linear equation is a linear method that is high-order accurate for smooth solutions, so by Godunov's theorem it **cannot be monotonicity preserving**. Any fix (limiters, etc.) must be **nonlinear** (solution-dependent).
### Homework 3: monotone fluxes
- **Godunov flux is monotone** for scalar convex $F$: using $\widehat F=\max\{F(\max(u_L,u^*)),F(\min(u_R,u^*))\}$ with $F$ decreasing for $u<u^*$ and increasing for $u>u^*$; the first term is non-decreasing in $u_L$, the second non-increasing in $u_R$; and the max of those is $(\uparrow,\downarrow)$. (Or use the $\min/\max$ over $[u_L,u_R]$ form.)
- **LLF flux is monotone** if $\alpha\ge\max|F'|$ over the relevant range:
  $$\partial_{u_L}\widehat F=\tfrac12\big(F'(u_L)+\alpha\big)\ge0,\qquad \partial_{u_R}\widehat F=\tfrac12\big(F'(u_R)-\alpha\big)\le0$$
- Coding (skipped): LLF for linear acoustics with $A=\begin{pmatrix}u_0&K_0\\1/\rho_0&u_0\end{pmatrix}$, $u_0=0$, $\rho_0=\tfrac14$, $K=2$, unknowns $(v,p)$, $\Delta t=h/c$, outflow BCs by setting the ghost state to $0$.
## 6. DG for elliptic problems (L6)
### Model problem and flux formulation
- Goal: DG discretization of $-\epsilon\Delta u$ (advection-diffusion-reaction $\nabla\cdot(\beta u)-\epsilon\Delta u+cu=f$ then combines with the earlier DG terms). Model problem: $-\Delta u=f$ in $\Omega$, $u=0$ on $\partial\Omega$.
- First-order form: $\sigma=\nabla u$ (flux), $-\nabla\cdot\sigma=f$ (potential $u$). Test with $\tau\in\Sigma_h$ and $v\in V_h$, integrate by parts per element, replace traces by numerical fluxes $\widehat u,\widehat\sigma$:
  $$\int_\Omega\sigma\cdot\tau\,dx=-\int_\Omega u\,\nabla_h\cdot\tau\,dx+\int_\Gamma\widehat u\,[\![\tau]\!]\,ds\qquad(F1)$$
  $$\int_\Omega\sigma\cdot\nabla_hv\,dx-\int_\Gamma\widehat\sigma\cdot[\![v]\!]\,ds=\int_\Omega fv\,dx\qquad(F2)$$
### Primal formulation (eliminate $\sigma$)
- Integrate (F1) by parts per element using the identity in [[#0. Notation and standard tools]], choose $\tau=\nabla_hv$, substitute in (F2):
  $$\int_\Omega\nabla_hu\cdot\nabla_hv-\int_\Gamma\big([\![u]\!]\cdot\{\nabla_hv\}+\{u\}[\![\nabla_hv]\!]\big)+\int_\Gamma\widehat u\,[\![\nabla_hv]\!]-\int_\Gamma\widehat\sigma\cdot[\![v]\!]=\int_\Omega fv\qquad(\text{DG primal form})$$
- **Lifting operators** $\ell:L^2(\Gamma)\to\Sigma_h$ and $r:[L^2(\Gamma)]^d\to\Sigma_h$:
  $$\int_\Omega\ell(q)\cdot\tau=-\int_\Gamma q[\![\tau]\!],\qquad\int_\Omega r(\varphi)\cdot\tau=-\int_\Gamma\varphi\cdot\{\tau\}\quad\forall\tau\in\Sigma_h$$
- Discrete flux in terms of $u$ and the numerical flux $\widehat u$:
  $$\sigma_h=\nabla_hu_h+r([\![u_h]\!])+\ell(\{u_h\}-\widehat u)$$
### Examples of DG methods (Arnold-Brezzi-Cockburn-Marini 2002 unified analysis)
- **BR1** (Bassi-Rebay): $\widehat u=\{u\}$, $\widehat\sigma=\{\sigma\}$. Simple, **not stable** (no penalty).
- **BR2:** BR1 modified with a lifted-jump stabilization; stable for a penalty constant given by a simple integer determined by mesh topology (number of faces per element).
- **Interior penalty (IP):** $\widehat u=\{u\}$, $\widehat\sigma=\{\nabla_hu\}-\alpha[\![u]\!]$. BR1 fixed by using $\{\nabla_hu\}$ and adding the penalty $\alpha[\![u]\!]$. Stable only for $\alpha$ large enough, $\alpha\sim h^{-1}$.
- **LDG** (local DG): with a prescribed vector $\boldsymbol\beta$,
  $$\widehat u=\{u\}-\boldsymbol\beta\cdot[\![u]\!],\qquad\widehat\sigma=\{\sigma\}+\boldsymbol\beta[\![\sigma]\!]-\alpha[\![u]\!]$$
For $|\boldsymbol\beta\cdot n|=\tfrac12$: $\widehat u$ is an "upwind" value (with arbitrary prescribed direction) and $\widehat\sigma$ the "downwind" value plus penalty. Stable for **any $\alpha>0$** (and for special $\boldsymbol\beta$, even $\alpha=0$: "minimal dissipation LDG"). Natural DG discretization of the first-order system.
### Symmetric interior penalty (SIPG) bilinear form
$$a(u,v)=\int_\Omega\nabla_hu\cdot\nabla_hv\,dx-\int_\Gamma[\![u]\!]\cdot\{\nabla_hv\}\,ds-\int_\Gamma\{\nabla_hu\}\cdot[\![v]\!]\,ds+\int_\Gamma\alpha\,[\![u]\!]\cdot[\![v]\!]\,ds$$
- Find $u_h\in V_h$ with $a(u_h,v_h)=(f,v_h)$ for all $v_h\in V_h$.
- Terms: consistency term, symmetry term (adjoint consistency), penalty term (stability). Symmetric, so the matrix is SPD when stable.
- Properties: **consistent** (exact $u$ satisfies $a(u,v)=(f,v)$ for all $v\in H^1(\mathcal T)$ with suitable regularity), so **Galerkin orthogonality** holds; **adjoint consistent**, enabling optimal $L^2$ error.
### Analysis of SIPG
- DG norm:
  $$|||v|||^2:=\|\nabla_hv\|^2_{L^2(\Omega)}+h^{-1}\|[\![v]\!]\|^2_{L^2(\Gamma)}$$
- Goals: **continuity** $a(u,v)\lesssim|||u|||\,|||v|||$; **coercivity** $a(u,u)\gtrsim|||u|||^2$; accuracy
  $$|||u-u_h|||\lesssim h^p|u|_{H^{p+1}},\qquad\|u-u_h\|_{L^2}\lesssim h^{p+1}|u|_{H^{p+1}}$$
- **Boundedness:** the problematic term $\int_\Gamma[\![u]\!]\cdot\{\nabla_hv\}=-\int_\Omega\nabla_hv\cdot r([\![u]\!])$. Cauchy-Schwarz plus the lifting bound
  $$\|r([\![u]\!])\|_{L^2(\Omega)}\lesssim h^{-1/2}\|[\![u]\!]\|_{L^2(\Gamma)}\quad(\text{trace inequality})$$
gives $\int_\Gamma[\![u]\!]\cdot\{\nabla_hv\}\lesssim h^{-1/2}\|[\![u]\!]\|_{L^2(\Gamma)}\|\nabla_hv\|_{L^2}$.
- **Coercivity:** $a(u,u)=\|\nabla_hu\|^2+\alpha\|[\![u]\!]\|^2-2\int_\Gamma\{\nabla_hu\}\cdot[\![u]\!]$ and the cross term is bounded using Young's inequality ($\epsilon=1/(2C)$):
  $$2\int_\Gamma\{\nabla_hu\}\cdot[\![u]\!]\le\tfrac12\|\nabla_hu\|^2+2C^2h^{-1}\|[\![u]\!]\|^2$$
So SIPG is stable iff $\alpha\gtrsim h^{-1}$ with a **large enough constant** (write $\alpha=\eta h^{-1}$, $\eta$ above a threshold depending on $C$ and the trace constant).
- Drawbacks: the threshold $\eta$ is not explicit a priori; the condition number grows with $\eta$. Fixes: (1) preconditioners robust w.r.t. $\alpha$ (see L7), (2) an explicit expression for the smallest stable $\alpha$ (Shahbazi 2005). LDG and BR2 avoid the issue.
- **Accuracy in the DG norm:** coercivity + Galerkin orthogonality + continuity give **Cea's lemma**
  $$|||u-u_h|||\lesssim|||u-v_h|||\quad\forall v_h\in V_h$$
Take $v_h=u_I$ the nodal interpolant (continuous, so $[\![u-u_I]\!]=0$ and $|||u-u_I|||=|u-u_I|_{H^1}$):
$$|||u-u_h|||\lesssim h^p|u|_{H^{p+1}(\Omega)}$$
- **Accuracy in $L^2$ (duality):** $-\Delta\psi=u-u_h$, $\psi=0$ on $\partial\Omega$. By consistency $a(\psi,v)=(u-u_h,v)$, then
  $$\|u-u_h\|^2=a(\psi-\psi_I,u-u_h)\lesssim|||\psi-\psi_I|||\,|||u-u_h|||\lesssim h|\psi|_{H^2}\,h^p|u|_{H^{p+1}}$$
and elliptic regularity $|\psi|_{H^2}\lesssim\|u-u_h\|$ gives $\|u-u_h\|_{L^2}\lesssim h^{p+1}|u|_{H^{p+1}}$. **Optimal** in both norms.
### Homework 3 (interior penalty results)
- **Jump of the DG solution:** 
  $$\|[\![u_h]\!]\|_{L^2(\Gamma)}\lesssim h^{p+1/2}|u|_{H^{p+1}(\Omega)}$$
Idea: $[\![u]\!]=0$ for the exact solution, so $[\![u_h]\!]=[\![u_h-u]\!]$ and $h^{-1/2}\|[\![u_h-u]\!]\|\le|||u-u_h|||\lesssim h^p|u|_{H^{p+1}}$.
- **Flux error** for the (potential, flux) IP-DG solution $(u_h,\sigma_h)$ with $\sigma=\nabla u$:
  $$\|\sigma-\sigma_h\|_{L^2(\Omega)}\lesssim h^p|u|_{H^{p+1}(\Omega)}$$
Idea: $\sigma_h=\nabla_hu_h+r([\![u_h]\!])$ (here $\widehat u=\{u\}$ kills the $\ell$ term), then triangle inequality and the lifting bound $\|r([\![u_h]\!])\|\lesssim h^{-1/2}\|[\![u_h]\!]\|\lesssim|||u-u_h|||$.
- **Trace inequality** (proved in HW3 Q6): $\|u_h\|_{L^2(\partial\kappa)}\lesssim h^{-1/2}\|u_h\|_{L^2(\kappa)}$ for $u_h\in V_h$ (scaling to reference triangle + equivalence of norms on $P_p(\hat\kappa)$).
## 7. Preconditioning DG (L7)
### Setup
- IP form with $\alpha=\eta h^{-1}$, $\eta$ bounded below. Matrix $A_{ij}=a(\varphi_i,\varphi_j)$ is SPD. Same DG norm and continuity/coercivity as in L6.
- **Lemma (Poincare-type, uses duality):** for $u\in H^1(\mathcal T)$,
  $$\|u\|^2_{L^2(\Omega)}\lesssim\|\nabla_hu\|^2_{L^2}+h^{-1}\|[\![u]\!]\|^2_{L^2(\Gamma)}=|||u|||^2$$
- **Inverse and trace inequalities** (L7 Prop. 1): $\|u_h\|_{H^1(\mathcal T)}\lesssim h^{-1}\|u_h\|_{L^2}$, $\|u_h\|_{L^2(\partial\kappa)}\lesssim h^{-1/2}\|u_h\|_{L^2(\kappa)}$.
- **Proposition 2:** for $u_h\in V_h$,
  $$\|u_h\|^2_{L^2(\Omega)}\lesssim a(u_h,u_h)\lesssim\eta\,h^{-2}\|u_h\|^2_{L^2(\Omega)}$$
(lower bound: coercivity + Poincare; upper bound: continuity + inverse inequality + trace inequality for jumps).
### Condition number
- Basis: Lagrange basis on each element, **not glued** (distinct basis functions per element, $\varphi_i$ supported on one element). DG allows interior nodes (no continuity needed).
- Use $\|u_h\|^2_{L^2}\approx h^d\|\boldsymbol u\|^2_{\ell^2}$ ($d=2$: $h^2$) in the Rayleigh quotient.
- **Theorem:**
  $$\lambda_{\min}(A)\approx h^2,\quad\lambda_{\max}(A)\approx\eta,\qquad\kappa(A)\approx\eta\,h^{-2}$$
- Same $h^{-2}$ conditioning as the $H^1$-conforming method; **worse as penalty $\eta$ increases**.
### Additive subspace correction (domain decomposition) preconditioner
- Decompose $V_h=V_0+V_1+\dots+V_N$.
- Coarse space $V_0=V_h\cap H^1(\Omega)$ (the conforming FE space on the same mesh).
- Local spaces $V_i=\mathrm{span}\{\varphi_i\}$ (one DG basis function each, so this is a block-Jacobi/point smoother).
- Preconditioner $B=\sum_{i=0}^NI_iA_i^{-1}I_i^T$, with $A_0^{-1}$ replaced by any good preconditioner $B_0$ for $A_0$ with $\kappa(B_0A_0)\lesssim1$ (two-level DD, multigrid, ...).
- Goal: bound the spectrum of $T=BA$. Two conditions (from MTH 652 Lecture 14):
- **(C1) Finite overlap:** $a(v_i,v_j)=0$ for all but a bounded number of $(i,j)$, $i,j>0$ (true: volume term couples only same-element functions, face terms only same-edge functions). Gives $\lambda_{\max}(T)\lesssim1$.
- **(C2) Stable decomposition:** for all $u_h$ there exists $u_h=u_0+\dots+u_N$ with $\sum_ia(u_i,u_i)\lesssim a(u_h,u_h)$. Gives $\lambda_{\min}(T)\gtrsim1$, via
  $$(T^{-1}v,v)_A=\inf_{\sum v_j=v}\sum_j(T_j^{-1}v_j,v_j)_A$$
- Construction of the decomposition: $u_0=Q_0u_h$ with $Q_0$ the **vertex-averaging operator** (conforming function whose value at each vertex is the average of the traces of $u_h$ from the incident triangles). Then $u_i$ ($i\ge1$) are the unique components of $u_h-u_0$ in the direct sum $V_1\oplus\dots\oplus V_N$. Key estimate
  $$\|u_h-Q_0u_h\|^2_{L^2(\Omega)}\lesssim h\|[\![u_h]\!]\|^2_{L^2(\Gamma)}$$
(zero if $u_h$ is continuous) yields $|||u_0|||\lesssim|||u_h|||$ and $\sum_{i\ge1}|||u_i|||^2\lesssim\eta h^{-1}\|[\![u_h]\!]\|^2\lesssim|||u_h|||^2$.
- **Conclusion:** the additive preconditioner is **robust**: $\kappa(BA)\lesssim1$ independent of $h$ and the penalty $\eta$, reducing DG preconditioning to the preconditioning of the $H^1$-conforming problem.
## 8. H(div) and H(curl) (L8)
### Motivation
- **Darcy flow** (velocity $u$, pressure $p$):
  $$u+\nabla p=f,\qquad\nabla\cdot u=g$$
Variational form: $(u,v)-(p,\nabla\cdot v)=(f,v)$, $(\nabla\cdot u,q)=(g,q)$. No derivative on $p,q$ so $p,q\in L^2$. Only the divergence of $u,v$ is needed: $u,v\in[L^2]^d$ with $\nabla\cdot u\in L^2$ (strictly larger than $[H^1]^d$).
- **Maxwell:** time-harmonic Maxwell reduces to the curl-curl problem $\nabla\times(\nabla\times E)+\alpha E=f$. Integration by parts:
  $$\int_\Omega v\cdot(\nabla\times u)\,dx=\int_\Omega u\cdot(\nabla\times v)\,dx-\int_{\partial\Omega}(u\times n)\cdot v\,ds$$
With zero tangential trace ($E\times n=0$, $F\times n=0$) the variational form is
$$(\nabla\times E,\nabla\times F)+(\alpha E,F)=(f,F)$$
Needs $E,\nabla\times E\in[L^2]^d$, larger than $[H^1]^d$.
### Spaces
$$H(\mathrm{div},\Omega)=\{v\in[L^2]^d:\nabla\cdot v\in L^2\},\qquad H(\mathrm{curl},\Omega)=\{v\in[L^2]^d:\nabla\times v\in[L^2]^d\}$$
- $[H^1]^d\subset H(\mathrm{div}),H(\mathrm{curl})\subset[L^2]^d$ strictly. 2D curl is scalar: $\nabla\times v=\partial_xv_2-\partial_yv_1$.
### Continuity requirements for piecewise smooth functions
- Let $u$ be piecewise $C^1$ w.r.t. the mesh. Characterization via the weak derivative and element-wise integration by parts:
  - $u\in H^1(\Omega)\iff u$ is continuous, i.e. $[\![u]\!]=0$ on $\Gamma$.
  - $u\in H(\mathrm{div},\Omega)\iff[\![u]\!]=u^-\cdot n^-+u^+\cdot n^+\equiv0$: **normal continuity** (tangential jumps allowed).
  - $u\in H(\mathrm{curl},\Omega)\iff[\![u\times n]\!]=u^-\times n^-+u^+\times n^+\equiv0$: **tangential continuity** (normal jumps allowed).
- Weak divergence: $\int_\Omega(\nabla_w\cdot v)\phi=-\int_\Omega v\cdot\nabla\phi$. Weak curl: $\int_\Omega(\nabla_w\times v)\cdot\phi=\int_\Omega v\cdot(\nabla\times\phi)$.
- Core computation: $\int_\Omega(\nabla_h\cdot u)\phi=\int_\Gamma[\![u]\!]\phi-\int_\Omega u\cdot\nabla\phi$, so the weak divergence equals the broken one iff $\int_\Gamma[\![u]\!]\phi=0$ for all $\phi$.
- A vector field decomposes on a face as $\phi=\phi_\parallel+(\phi\cdot n)n$, and $\phi\times n=\phi_\parallel\times n$, so $[\![u\times n]\!]=0$ is tangential continuity.
- FE consequence: membership in $H^1$, $H(\mathrm{div})$, $H(\mathrm{curl})$ is decided by interface continuity of full / normal / tangential components.
## 9. Raviart-Thomas elements and Darcy (L9)
### Why not standard elements
- $H^1$ spaces are subspaces of $H(\mathrm{div})$ but are often **not suitable**. For Darcy, $V_h$ = continuous piecewise linears (vector) and $P_h$ = piecewise constants is **not inf-sup stable**: $\dim V_h=2n_V<n_T=\dim P_h$ (angle counting), so $B$ ($n_T\times2n_V$) has dependent rows and some $q_h\ne0$ has $b(v_h,q_h)=0$ for all $v_h$. Cause: $V_h$ is too small; the flux should be in $H(\mathrm{div})$ not $H^1$. We need something "between" $H^1$ (full continuity) and DG (no continuity).
### Lowest-order Raviart-Thomas on a triangle
$$RT_0(\kappa)=\{v(x)=a+bx:\ a\in\mathbb R^2,\ b\in\mathbb R\}\quad(\dim=3;\ \text{in }\mathbb R^d:\ \dim=d+1)$$
- $\nabla\cdot v=2b$ (in $d$ dims: $db$). So $\nabla\cdot v=0\Rightarrow v$ constant.
- **Normal component is constant on each edge:** the edge lies on $n\cdot x=c$, so $v\cdot n=a\cdot n+bc$.
- **Degrees of freedom** (moments of normal flux): $c_i=\int_{e_i}v\cdot n\,ds$, $i=1,2,3$. **Unisolvent**: if all $c_i=0$ then $0=\sum c_i=\int_{\partial\kappa}v\cdot n=\int_\kappa\nabla\cdot v=2|\kappa|b$, so $b=0$; then $c_i=|e_i|a\cdot n_i=0$ for three independent normals gives $a=0$. Injective linear map between 3-dimensional spaces, hence bijective.
- DOFs induce a dual basis $\phi_i$, $c_i(\phi_j)=\delta_{ij}$.
- **Global space:** 
  $$V_h=\{v\in H(\mathrm{div},\Omega):v|_\kappa\in RT_0(\kappa)\}$$
One DOF per mesh edge; the global basis function for an edge is supported on the two triangles sharing it (normal continuity built in). Contains some but not all piecewise linears.
### Piola transformation (normal-preserving)
- Affine map $T:\hat\kappa\to\kappa$, $J=DT$. $H^1$ basis functions are just composed with $T^{-1}$, but this **does not** preserve normal-flux DOFs for $H(\mathrm{div})$.
- Want $\int_{\partial\kappa}w\,v\cdot n=\int_{\partial\hat\kappa}\hat w\,\hat v\cdot\hat n$ with $\hat w=w\circ T$. Suffices $\int_\kappa v\cdot\nabla w=\int_{\hat\kappa}\hat v\cdot\nabla\hat w$ and $\int_\kappa(\nabla\cdot v)w=\int_{\hat\kappa}(\nabla\cdot\hat v)\hat w$.
- Result (**contravariant Piola transform**):
  $$\hat v=\det(J)\,J^{-1}\,v\circ T,\qquad v=\frac1{\det J}\,J\,\hat v\circ T^{-1},\qquad\nabla\cdot\hat v=\det(J)\,(\nabla\cdot v)\circ T$$
- HW4 Q1: the divergence identity $\int_\kappa(\nabla\cdot v)w\,dx=\int_{\hat\kappa}(\nabla\cdot\hat v)\hat w\,d\hat x$ follows from $\nabla\cdot\hat v=\det(J)(\nabla\cdot v)\circ T$ and change of variables $dx=|\det J|d\hat x$ (equivalently, from the gradient identity and the divergence theorem on both sides).
### RT interpolant and commuting diagram
- $I_h:[C^1]^d\to V_h$ defined by matching edge fluxes:
  $$\int_ev_h\cdot n\,ds=\int_ev\cdot n\,ds\ \ \forall e\iff I_hv=\sum_{e\in\Gamma}\Big(\int_ev\cdot n\,ds\Big)\phi_e$$
Extends to $[H^1]^d$ (trace theorem). $I_h$ is a projection onto $V_h$.
- Let $P_h$ = piecewise constants and $\Pi_0$ the $L^2$ projection onto $P_h$. For $w_h\in P_h$, $\int_\Omega(\nabla\cdot I_hv)w_h=\sum_\kappa\int_{\partial\kappa}v\cdot n\,w_h=\int_\Omega(\nabla\cdot v)w_h$. Hence
  $$\nabla\cdot I_hv=\Pi_0(\nabla\cdot v)$$
- **Commuting diagram:** $[H^1]^d\xrightarrow{\nabla\cdot}L^2$ over $V_h\xrightarrow{\nabla\cdot}P_h$ with vertical maps $I_h$ and $\Pi_0$.
- **Lemma 1:** $\|I_hv\|_{L^2}\lesssim\|v\|_{L^2}+h\|\nabla v\|_{L^2}$ and $\|I_hv\|_{H(\mathrm{div})}\lesssim\|v\|_{H^1}$ (Piola + trace theorem + shape regularity $\|J\|\sim h$, $\|J^{-1}\|\sim h^{-1}$).
- Approximation: $\|v-I_hv\|_{L^2}\lesssim h\|v\|_{H^1}$ (reduces to approximation by $P_0$ since $I_h$ is a projection and $P_0\subset RT_0$; for $RT_k$ reduces to $P_k$).
### Application to Darcy's equations (saddle point problem)
- With $V=H(\mathrm{div},\Omega)$, $P=L^2(\Omega)$:
  $$a(u,v)=(u,v),\quad b(v,p)=-(\nabla\cdot v,p),\quad F(v)=(f,v),\quad G(q)=(-g,q)$$
  $$a(u,v)+b(v,p)=F(v),\qquad b(u,q)=G(q)$$
- **Brezzi well-posedness conditions** (MTH 652):
  - (C1) continuity of forms.
  - (C2) $a$ coercive on $Z=\ker b=\{v:\nabla\cdot v=0\}$. Here $a(v,v)=\|v\|^2_{L^2}=\|v\|^2_{H(\mathrm{div})}$ on $Z$, constant 1.
  - (C3) **inf-sup:**
    $$\inf_{q\in P}\sup_{v\in V}\frac{b(v,q)}{\|v\|_V\|q\|_P}\ge\beta>0$$
Proof: given $q$ solve $-\Delta w=q$, $\|w\|_{H^2}\lesssim\|q\|$, take $v=\nabla w$ so $\nabla\cdot v=-q$, $\|v\|_{H^1}\lesssim\|q\|$.
- **Discrete inf-sup:** continuous $P_1$ / $P_0$ fails (above). **$RT_0$-$P_0$ is inf-sup stable.** Proof: take $v$ from the continuous proof, set $v_h=I_hv$, so $\nabla\cdot v_h=\Pi_0(\nabla\cdot v)=-q_h$ (as $q_h\in P_h$), and $\|v_h\|_{H(\mathrm{div})}\lesssim\|v\|_{H^1}\lesssim\|q_h\|$ (Lemma 1, a **Fortin operator** argument).
- **Convergence (general theory):**
  $$\|u-u_h\|_{H(\mathrm{div})}\le C_1\inf_{v_h}\|u-v_h\|_{H(\mathrm{div})}+C_2\inf_{q_h}\|p-q_h\|_{L^2}$$
  $$\|p-p_h\|_{L^2}\le C_3\inf_{v_h}\|u-v_h\|_{H(\mathrm{div})}+C_4\inf_{q_h}\|p-q_h\|_{L^2}$$
- **Improved flux estimate:** $b(u-u_h,q_h)=0$ for all $q_h\in P_h$ and $b(u-I_hu,q_h)=0$ give $\nabla\cdot u_h=\nabla\cdot I_hu=\Pi_0\nabla\cdot u$. Taking $v_h=I_hu-u_h$ in the first equation, $(u-u_h,I_hu-u_h)=0$ (an $L^2$ orthogonality), hence
  $$\|u-u_h\|_{L^2}\le\|u-I_hu\|_{L^2}\lesssim h\|u\|_{H^1}$$
The velocity $L^2$ error depends only on the interpolation error of $u$ (**independent of $p$**).
### Homework 4: Raviart-Thomas in 3D
- $RT_0(\kappa)=\{a+bx:a\in\mathbb R^3,b\in\mathbb R\}$ on a tetrahedron, $\dim=4$.
- (a) $\nabla\cdot v=3b$ (in general $d\,b$).
- (b) Normal component constant on each face: face lies in the plane $n\cdot x=c$, so $v\cdot n=a\cdot n+bc$.
- (c) DOFs: $c_i=\int_{f_i}v\cdot n\,ds$, one per face ($4$ faces, matching $\dim=4$).
- (d) Unisolvence: $\sum_ic_i=\int_{\partial\kappa}v\cdot n=3|\kappa|b$, so $c_i=0\Rightarrow b=0$; then $|f_i|\,a\cdot n_i=0$ for the four face normals, which span $\mathbb R^3$, so $a=0$. Injective map between 4-dim spaces is bijective.
- (e) Global space: $V_h=\{v\in H(\mathrm{div}):v|_\kappa\in RT_0\}$ with one global DOF (flux through the face, with a fixed global orientation of the normal) per mesh face; global basis function supported on the two tetrahedra sharing the face; transfer to physical elements via the Piola transform $v=\frac1{\det J}J\hat v\circ T^{-1}$. Normal continuity then holds automatically, since the normal component on a shared face is determined by the single shared DOF.
## 10. Nedelec elements and the de Rham complex (L10)
### 2D Nedelec ($H(\mathrm{curl})$) elements
- In 2D, $\nabla\times u=\nabla\cdot(u^\perp)$ with $u^\perp=(-u_2,u_1)$. So 2D $H(\mathrm{curl})$ elements are $90^\circ$ **rotations** of $H(\mathrm{div})$ elements (2D only!).
  $$N_0(\kappa)=\{u=a+b\,x^\perp:a\in\mathbb R^2,b\in\mathbb R\},\quad x^\perp=(-y,x)$$
- $\nabla u=\begin{pmatrix}0&-b\\b&0\end{pmatrix}$ (skew), $\nabla\times u=2b$, so curl-free implies constant.
- **Tangential component is constant on each edge:** $u\cdot t=a\cdot t+bc$ on the line $n\cdot x=c$, $t=n^\perp$.
- **DOFs:** $c_e(u)=\int_eu\cdot t_e\,ds$ per edge (with orientation). Unisolvent (same argument as $RT_0$). $\dim N_0=3$.
- **Piola transform (covariant, tangential-preserving):**
  $$\hat v=J^T\,v\circ T,\qquad\phi=J^{-T}\,\hat\phi\circ T^{-1}$$
preserves $\int_{\partial\kappa}w\,v\cdot t$.
- Global space: $V_h=\{v\in H(\mathrm{curl},\Omega):v|_\kappa\in N_0(\kappa)\}$, one oriented global DOF $c_e$ per edge.
- **Interpolant** $I_hv$: $\int_eI_hv\cdot t=\int_ev\cdot t$ for all $e$. Using the curl integration by parts formula $\int_\kappa w\nabla\times v=-\int_\kappa\nabla w\cdot v^\perp-\int_{\partial\kappa}wv\cdot t$,
  $$\nabla\times I_hv=\Pi_0(\nabla\times v)$$
(commuting diagram with $P_h$ = piecewise constants).
### Relation to $H^1$ (2D)
- $V_{H^1}$ = continuous piecewise linears, $V_N$ = lowest-order Nedelec, $V_{L^2}=P_h$. For $v\in V_{H^1}$, $\nabla v$ is elementwise constant (in $N_0$) with tangential continuity, so $\nabla v\in V_N$.
- **Commuting diagram:**
  $$C^2\xrightarrow{\nabla}C^1\xrightarrow{\nabla\times}C^0\quad\text{over}\quad V_{H^1}\xrightarrow{\nabla}V_N\xrightarrow{\nabla\times}V_{L^2}$$
with vertical maps $I_{H^1},I_N,\Pi_0$.
- Left square proof: $\int_e\nabla v_h\cdot t=[v_h]_{\partial e}=[v]_{\partial e}=\int_e\nabla v\cdot t$ (fundamental theorem of calculus), so DOFs of $\nabla I_{H^1}v$ and $I_N\nabla v$ agree; unisolvence gives $\nabla I_{H^1}v=I_N\nabla v$.
- Same argument gives the RT diagram: $V_{H^1}\xrightarrow{\nabla^\perp}V_{RT}\xrightarrow{\nabla\cdot}V_{L^2}$.
### 3D Nedelec
- In 3D $RT$ DOFs live on **faces**, Nedelec DOFs on **edges** (spaces are genuinely different, not rotations).
  $$N_0=\{v=a+q:\ q\in S\},\qquad S=\mathrm{span}\Big\{\begin{pmatrix}0\\-z\\y\end{pmatrix},\begin{pmatrix}z\\0\\-x\end{pmatrix},\begin{pmatrix}-y\\x\\0\end{pmatrix}\Big\}=\{b\times x:b\in\mathbb R^3\}$$
- $\dim N_0=6$ (3 for $a$, 3 for $q$), one DOF per edge (6 edges of a tetrahedron): $c_e(v)=\int_ev\cdot t\,ds$. Unisolvence is harder than in 2D but holds.
- $q\in S$ satisfies $q(x)\cdot x=0$. $\nabla v$ is **skew-symmetric**. $\nabla\times v=0\iff\nabla v$ symmetric, so curl-free implies constant. Tangential component is constant along every line.
- **3D commuting diagram (discrete de Rham complex):**
  $$C^3\xrightarrow{\nabla}C^2\xrightarrow{\nabla\times}C^1\xrightarrow{\nabla\cdot}C^0\quad\text{over}\quad V_{H^1}\xrightarrow{\nabla}V_N\xrightarrow{\nabla\times}V_{RT}\xrightarrow{\nabla\cdot}V_{L^2}$$
with vertical maps $I_{H^1},I_N,I_{RT},\Pi_0$.
- Footnote: on quadrilaterals/hexahedra the DOFs are the same type (normal/tangential integrals on edges/faces) but there are more edges/faces so the local space must be larger.
## 11. Quick reference: rates, key formulas, exam checklist
### Convergence rates and norm results
- $H^1$-conforming FEM for advection: $\|u-u_h\|_{L^2}\lesssim h|u|_{H^2}$ (suboptimal by one order), needs $u\in H^2$.
- DG advection, upwind ($b_0$ bounded below): $|||u-u_h|||_{b_0}\lesssim h^{p+1/2}|u|_{H^{p+1}}$.
- DG advection, central ($b_0=0$): $|||u-u_h|||_{b_0}\lesssim h^p|u|_{H^{p+1}}$.
- SIPG Poisson: $|||u-u_h|||\lesssim h^p|u|_{H^{p+1}}$, $\|u-u_h\|_{L^2}\lesssim h^{p+1}|u|_{H^{p+1}}$, $\|[\![u_h]\!]\|_{L^2(\Gamma)}\lesssim h^{p+1/2}|u|_{H^{p+1}}$, $\|\sigma-\sigma_h\|_{L^2}\lesssim h^p|u|_{H^{p+1}}$. Penalty $\alpha=\eta/h$, $\eta$ large enough.
- SIPG matrix: $\kappa(A)\approx\eta h^{-2}$. With the additive subspace preconditioner (conforming coarse space + point blocks): $\kappa(BA)\lesssim1$, independent of $h,\eta$.
- $RT_0$-$P_0$ for Darcy: inf-sup stable; $\|u-u_h\|_{L^2}\le\|u-I_hu\|_{L^2}\lesssim h\|u\|_{H^1}$, $\nabla\cdot u_h=\Pi_0\nabla\cdot u$.
### Key formulas
- Rankine-Hugoniot: $\sigma=\dfrac{F(u_L)-F(u_R)}{u_L-u_R}$. Burgers: $\sigma=\tfrac12(u_L+u_R)$.
- Entropy: $F'(u_L)>\sigma>F'(u_R)$ (shock if $u_L>u_R$ for convex $F$; rarefaction if $u_L<u_R$).
- Rarefaction: $u=(F')^{-1}(x/t)$ on $F'(u_L)<x/t<F'(u_R)$.
- Characteristics: $x=s+F'(u_0(s))t$; Burgers breaking time $t^*=-1/\min u_0'$.
- Numerical flux conditions (DG advection/conservation laws): consistent $\widehat F(u,u)=F(u)$; conservative (single-valued); monotone for FV monotonicity ($\widehat F(\uparrow,\downarrow)$).
- LLF: $\widehat F=\tfrac12(F(u_L)+F(u_R))+\tfrac\alpha2(u_L-u_R)$, $\alpha\ge$ max wave speed; adds artificial viscosity $\epsilon=h\alpha/2$.
- Godunov flux $=F(w^*)$ (exact Riemann); upwind for linear advection.
- DG advection coercivity: $a(u,u)=\|c_0u\|^2+\int_\Gamma b_0|\beta\cdot n|[\![u]\!]^2$.
- Forward Euler stability region: $|1+\lambda\Delta t|\le1$.
- RT: $v=a+bx$, DOFs $\int_e v\cdot n$, $\nabla\cdot I_h v=\Pi_0\nabla\cdot v$, Piola $v=\frac1{\det J}J\hat v\circ T^{-1}$. Nedelec: $v=a+bx^\perp$, DOFs $\int_e v\cdot t$, $\nabla\times I_hv=\Pi_0\nabla\times v$, Piola $\phi=J^{-T}\hat\phi\circ T^{-1}$.
### Conceptual questions to be able to answer
- Why does the central/$H^1$-conforming discretization of advection oscillate, and what does upwinding fix? (Directionality of characteristics; stability region of forward Euler; extra jump control in the DG norm.)
- Define consistent and conservative numerical fluxes. Why is the DG formulation consistent iff the flux is consistent?
- Why do shocks form from smooth data? Why are weak solutions non-unique? How does the entropy (vanishing viscosity) condition pick the physical one?
- Derive the Riemann solution for Burgers (shock vs fan), the Godunov flux, and the LLF flux. Compare accuracy/cost.
- State Godunov's theorem and what it implies for high-order DG.
- Derive the SIPG form from the flux formulation; why $\alpha\sim h^{-1}$; why boundedness uses lifting operators/trace inequality; why $L^2$ error is optimal (duality, adjoint consistency).
- Compare SIPG, LDG, BR1, BR2.
- Why does $\kappa(A)\sim\eta h^{-2}$ and how does subspace correction with a conforming coarse space give robustness?
- Why do $H(\mathrm{div})$ and $H(\mathrm{curl})$ require normal / tangential continuity? Construct $RT_0$ and $N_0$, show unisolvence, and explain Piola transforms.
- Why does the commuting-diagram property give discrete inf-sup stability for Darcy ($RT_0$-$P_0$) but $P_1$-$P_0$ fails?
