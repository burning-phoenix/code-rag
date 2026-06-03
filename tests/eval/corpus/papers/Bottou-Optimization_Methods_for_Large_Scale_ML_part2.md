

---

## Page 1

### 5.3.3 Commentary

Although the gradient aggregation methods discussed in this section enjoy a faster rate of convergence than SG (i.e., linear vs. sublinear), they should not be regarded as clearly superior to SG. After all, following similar analysis as in §4.4, the computing time for SG can be shown to be  $T(n, \epsilon) \sim \kappa^2/\epsilon$  with  $\kappa := L/c$ . (In fact, a computing time of  $\kappa/\epsilon$  is often observed in practice.) On the other hand, the computing times for SVRG, SAGA, and SAG are

$$\mathcal{T}(n, \epsilon) \sim (n + \kappa) \log(1/\epsilon),$$

which grows with the number of examples  $n$ . Thus, following similar analysis as in §4.4, one finds that, for very large  $n$ , gradient aggregation methods are comparable to batch algorithms and therefore cannot beat SG in this regime. For example, if  $\kappa$  is close to 1, then SG is clearly more efficient since within a single epoch it reaches the optimal testing error [25]. On the other hand, there exists a regime with  $\kappa \gg n$  in which gradient aggregation methods may be superior, and perhaps even easier to tune. At present, it is not known how useful gradient aggregation methods will prove to be in the future of large-scale machine learning. That being said, they certainly represent a class of optimization methods of interest due to their clever use of past information.

## 5.4 Iterate Averaging Methods

Since its inception, it has been observed that SG generates noisy iterate sequences that tend to oscillate around minimizers during the optimization process. Hence, a natural idea is to compute a corresponding sequence of *iterate averages* that would automatically possess less noisy behavior. Specifically, for minimizing a continuously differentiable  $F$  with unbiased gradient estimates, the idea is to employ the iteration

$$\begin{aligned} w_{k+1} &\leftarrow w_k - \alpha_k g(w_k, \xi_k) \\ \text{and } \tilde{w}_{k+1} &\leftarrow \frac{1}{k+1} \sum_{j=1}^{k+1} w_j, \end{aligned} \tag{5.16}$$

where the averaged sequence  $\{\tilde{w}_k\}$  has no effect on the computation of the SG iterate sequence  $\{w_k\}$ . Early hopes were that this auxiliary averaged sequence might possess better convergence properties than the SG iterates themselves. However, such improved behavior was found to be elusive when using classical stepsize sequences that diminished with a rate of  $\mathcal{O}(1/k)$  [124].

A fundamental advancement in the use of iterate averaging came with the work of Polyak [125], which was subsequently advanced with Juditsky [126]; see also the work of Ruppert [136] and Nemirovski and Yudin [106]. Here, the idea remains to employ the iteration (5.16), but with stepsizes diminishing at a slower rate of  $\mathcal{O}(1/(k^a))$  for some  $a \in (\frac{1}{2}, 1)$ . When minimizing strongly convex objectives, it follows from this choice that  $\mathbb{E}[\|w_k - w_*\|_2^2] = \mathcal{O}(1/(k^a))$  while  $\mathbb{E}[\|\tilde{w}_k - w_*\|_2^2] = \mathcal{O}(1/k)$ . What is interesting, however, is that in certain cases this combination of *long steps* and *averaging* yields an optimal constant in the latter rate (for the iterate averages) in the sense that no rescaling of the steps—through multiplication with a positive definite matrix (see (6.2) in the next section)—can improve the asymptotic rate or constant. This shows that, due to averaging, the adverse effects caused by ill-conditioning disappear. (In this respect, the effect of averaging has been characterized as being similar to that of using a second-order method in which the Hessian approximations approach the Hessian of the objective at the minimizer [125, 22]; see §6.2.1.) This


---

## Page 2

asymptotic behavior is difficult to achieve in practice, but is possible in some circumstances with careful selection of the stepsize sequence [161].

Iterate averaging has since been incorporated into various schemes in order to allow longer steps while maintaining desired rates of convergence. Examples include the *robust SA* and *mirror descent SA* methods presented in [105], as well as Nesterov’s *primal-dual averaging* method proposed in [109, §6]. This latter method is notable in this section as it employs gradient aggregation and yields a  $\mathcal{O}(1/k)$  rate of convergence for the averaged iterate sequence.

## 6 Second-Order Methods

In §5, we looked beyond classical SG to methods that are less affected by noise in the stochastic directions. Another manner to move beyond classical SG is to address the adverse effects of high nonlinearity and ill-conditioning of the objective function through the use of second-order information. As we shall see, these methods improve convergence rates of batch methods or the constants involved in the sublinear convergence rate of stochastic methods.

A common way to motivate second-order algorithms is to observe that first-order methods, such as SG and the full gradient method, are not *scale invariant*. Consider, for example, the full gradient iteration for minimizing a continuously differentiable function  $F : \mathbb{R}^d \rightarrow \mathbb{R}$ , namely

$$w_{k+1} \leftarrow w_k - \alpha_k \nabla F(w_k). \quad (6.1)$$

An alternative iteration is obtained by applying a full gradient approach after a linear transformation of variables, i.e., by considering  $\min_{\tilde{w}} F(B\tilde{w})$  for some symmetric positive definite matrix  $B$ . The full gradient iteration for this problem has the form

$$\tilde{w}_{k+1} \leftarrow \tilde{w}_k - \alpha_k B \nabla F(B\tilde{w}_k),$$

which, after scaling by  $B$  and defining  $\{w_k\} := \{B\tilde{w}_k\}$ , corresponds to the iteration

$$w_{k+1} \leftarrow w_k - \alpha_k B^2 \nabla F(w_k). \quad (6.2)$$

Comparing (6.2) with (6.1), it is clear that the behavior of the algorithm will be different under this change of variables. For instance, when  $F$  is a strongly convex quadratic with unique minimizer  $w_*$ , the full gradient method (6.1) generally requires many iterations to approach the minimizer, but from any initial point  $w_1$  the iteration (6.2) with  $B = (\nabla^2 F(w_1))^{-1/2}$  and  $\alpha_1 = 1$  will yield  $w_2 = w_*$ . These latter choices correspond to a single iteration of *Newton’s method* [52]. In general, it is natural to seek transformations that perform well in theory and in practice.

Another motivation for second-order algorithms comes from the observation that each iteration of the form (6.1) or (6.2) chooses the subsequent iterate by first computing the minimizer of a second-order Taylor series approximation  $q_k : \mathbb{R}^d \rightarrow \mathbb{R}$  of  $F$  at  $w_k$ , which has the form

$$q_k(w) = F(w_k) + \nabla F(w_k)^T (w - w_k) + \frac{1}{2} (w - w_k)^T B^{-2} (w - w_k). \quad (6.3)$$

The full gradient iteration corresponds to  $B^{-2} = I$  while Newton’s method corresponds to  $B^{-2} = \nabla^2 F(w_k)$  (assuming this Hessian is positive definite). Thus, in general, a full gradient iteration works with a model that is only first-order accurate while Newton’s method applies successive local re-scalings based on minimizing an exact second-order Taylor model of  $F$  at each iterate.


---

## Page 3

Deterministic (i.e., batch) methods are known to benefit from the use of second-order information; e.g., Newton’s method achieves a quadratic rate of convergence if  $w_1$  is sufficiently close to a strong minimizer [52]. On the other hand, stochastic methods like the SG method in §4 cannot achieve a convergence rate that is faster than sublinear, regardless of the choice of  $B$ ; see [1, 104]. (More on this in §6.2.1.) Therefore, it is natural to ask: can there be a benefit to incorporating second-order information in stochastic methods? We address this question throughout this section by showing that the careful use of successive re-scalings based on (approximate) second-order derivatives can be beneficial between the stochastic and batch regimes.

We begin this section by considering a *Hessian-free Newton* method that employs exact second-order information, but in a judicious manner that exploits the stochastic nature of the objective function. We then describe methods that attempt to mimic the behavior of a Newton algorithm through first-order information computed over sequences of iterates; these include *quasi-Newton*, *Gauss-Newton*, and related algorithms that employ only diagonal re-scalings. We also discuss the *natural gradient* method, which defines a search direction in the space of realizable distributions, rather than in the space of the real parameter vector  $w$ . Whereas Newton’s method is invariant to linear transformations of the variables, the natural gradient method is invariant with respect to more general invertible transformations.

We depict the methods of interest in this section on the downward axis illustrated in Figure 6.1. We use double-sided arrows for the methods that can be effective throughout the spectrum between the stochastic and batch regimes. Single-sided arrows are used for those methods that one might consider to be effective only with at least a moderate batch size in the stochastic gradient estimates. We explain these distinctions as we describe the methods.

The diagram illustrates the spectrum of optimization methods. A diagonal line labeled "Second-order methods" connects the "Stochastic gradient method" at the top to the "Stochastic Newton method" at the bottom. A horizontal line extends from the "Stochastic gradient method" to the "Batch gradient method". Dotted arrows indicate the effective regime of each method: "Diagonal Scaling", "quasi-Newton", and "Gauss-Newton" have double-headed arrows spanning the entire spectrum; "Hessian-free Newton" and "Natural gradient" have single-headed arrows pointing towards the "Batch gradient method".

Fig. 6.1: View of the schematic from Figure 3.3 with a focus on second-order methods. The dotted arrows indicate the effective regime of each method: the first three methods can employ mini-batches of any size, whereas the last two methods are efficient only for moderate-to-large mini-batch sizes.


---

## Page 4

## 6.1 Hessian-Free Inexact Newton Methods

Due to its scale invariance properties and its ability to achieve a quadratic rate of convergence in the neighborhood of a strong local minimizer, Newton’s method represents an ideal in terms of optimization algorithms. It does not scale well with the dimension  $d$  of the optimization problem, but there are variants that can scale well while also being able to deal with nonconvexity.

When minimizing a twice-continuously differentiable  $F$ , a Newton iteration is

$$w_{k+1} \leftarrow w_k + \alpha_k s_k, \quad (6.4a)$$

$$\text{where } s_k \text{ satisfies } \nabla^2 F(w_k) s_k = -\nabla F(w_k). \quad (6.4b)$$

This iteration demands much in terms of computation and storage. However, rather than solve the Newton system (6.4b) exactly through matrix factorization techniques, one can instead only solve it inexactly through an iterative approach such as the conjugate gradient (CG) method [69]. By ensuring that the linear solves are accurate enough, such an *inexact Newton-CG* method can enjoy a superlinear rate of convergence [47].

In fact, the computational benefits of inexact Newton-CG go beyond its ability to maintain classical convergence rate guarantees. Like many iterative linear system techniques, CG applied to (6.4b) does not require access to the Hessian itself, only Hessian-vector products [121]. It is in this sense that such a method may be called *Hessian-free*. This is ideal when such products can be coded directly without having to form an explicit Hessian, as Example 6.1 below demonstrates. Each product is at least as expensive as a gradient evaluation, but as long as the number of products—one per CG iteration—is not too large, the improved rate of convergence can compensate for the extra per-iteration work required over a simple full gradient method.

**Example 6.1.** Consider the function of the parameter vector  $w = (w_1, w_2)$  given by  $F(w) = \exp(w_1 w_2)$ . Let us define, for any  $d \in \mathbb{R}^2$ , the function

$$\phi(w; d) = \nabla F(w)^T d = w_2 \exp(w_1 w_2) d_1 + w_1 \exp(w_1 w_2) d_2.$$

Computing the gradient of  $\phi$  with respect to  $w$ , we have

$$\nabla_w \phi(w; d) = \nabla^2 F(w) d = \begin{bmatrix} w_2^2 \exp(w_1 w_2) d_1 + (\exp(w_1 w_2) + w_1 w_2 \exp(w_1 w_2)) d_2 \\ (\exp(w_1 w_2) + w_1 w_2 \exp(w_1 w_2)) d_1 + w_1^2 \exp(w_1 w_2) d_2 \end{bmatrix}.$$

We have thus obtained, for any  $d \in \mathbb{R}^2$ , a formula for computing  $\nabla^2 F(w) d$  that does not require  $\nabla^2 F(w)$  explicitly. Note that by storing the scalars  $w_1 w_2$  and  $\exp(w_1 w_2)$  from the evaluation of  $F$ , the additional costs of computing the gradient-vector and Hessian-vector products are small.

The idea illustrated in this example can be applied in general; e.g., see also Example 6.2 on page 54. For a smooth objective function  $F$ , one can compute  $\nabla^2 F(w) d$  at a cost that is a small multiple of the cost of evaluating  $\nabla F$ , and without forming the Hessian, which would require  $\mathcal{O}(d^2)$  storage. The savings in computation come at the expense of storage of some additional quantities, as explained in Example 6.1.

In machine learning applications, including those involving multinomial logistic regression and deep neural networks, Hessian-vector products can be computed in this manner, and an inexact Newton-CG method can be applied. A concern is that, in certain cases, the cost of the CG iterations may render such a method uncompetitive with alternative approaches, such as an SG method or a limited memory BFGS method (see §6.2), which have small computational overhead. Interestingly, however, the structure of the risk measures (3.1) and (3.2) can be exploited so that the resulting method has lighter computational overheads, as described next.


---

## Page 5

### 6.1.1 Subsampled Hessian-Free Newton Methods

The motivation for the method we now describe stems from the observation that, in inexact Newton methods, the Hessian matrix need not be as accurate as the gradient to yield an effective iteration. Translated to the context of large-scale machine learning applications, this means that the iteration is more tolerant to noise in the Hessian estimate than it is to noise in the gradient estimate.

Based on this idea, the technique we state here employs a smaller sample for defining the Hessian than for the stochastic gradient estimate. Following similar notation as introduced in §5.2, let the stochastic gradient estimate be

$$\nabla f_{\mathcal{S}_k}(w_k; \xi_k) = \frac{1}{|\mathcal{S}_k|} \sum_{i \in \mathcal{S}_k} \nabla f(w_k; \xi_{k,i})$$

and let the stochastic Hessian estimate be

$$\nabla^2 f_{\mathcal{S}_k^H}(w_k; \xi_k^H) = \frac{1}{|\mathcal{S}_k^H|} \sum_{i \in \mathcal{S}_k^H} \nabla^2 f(w_k; \xi_{k,i}), \quad (6.5)$$

where  $\mathcal{S}_k^H$  is conditionally uncorrelated with  $\mathcal{S}_k$  given  $w_k$ . If one chooses the *subsample* size  $|\mathcal{S}_k^H|$  small enough, then the cost of each product involving the Hessian approximation can be reduced significantly, thus reducing the cost of each CG iteration. On the other hand, one should choose  $|\mathcal{S}_k^H|$  large enough so that the curvature information captured through the Hessian-vector products is productive. If done appropriately, *Hessian subsampling* is robust and effective [2, 29, 122, 132]. An inexact Newton method that incorporates this techniques is outlined in Algorithm 6.1. The algorithm is stated with a backtracking (Armijo) line search [114], though other stepsize-selection techniques could be considered as well.

---

#### Algorithm 6.1 Subsampled Hessian-Free Inexact Newton Method

---

1. 1: Choose an initial iterate  $w_1$ .
2. 2: Choose constants  $\rho \in (0, 1)$ ,  $\gamma \in (0, 1)$ ,  $\eta \in (0, 1)$ , and  $\max_{cg} \in \mathbb{N}$ .
3. 3: **for**  $k = 1, 2, \dots$  **do**
4. 4: Generate realizations of  $\xi_k$  and  $\xi_k^H$  corresponding to  $\mathcal{S}_k$  and  $\mathcal{S}_k^H$ .
5. 5: Compute  $s_k$  by applying Hessian-free CG to solve

$$\nabla^2 f_{\mathcal{S}_k^H}(w_k; \xi_k^H) s = -\nabla f_{\mathcal{S}_k}(w_k; \xi_k) \quad (6.6)$$

until  $\max_{cg}$  iterations have been performed or a trial solution yields

$$\|r_k\|_2 := \|\nabla^2 f_{\mathcal{S}_k^H}(w_k; \xi_k^H) s + \nabla f_{\mathcal{S}_k}(w_k; \xi_k)\|_2 \leq \rho \|\nabla f_{\mathcal{S}_k}(w_k; \xi_k)\|_2.$$

1. 6: Set  $w_{k+1} \leftarrow w_k + \alpha_k s_k$ , where  $\alpha_k \in \{\gamma^0, \gamma^1, \gamma^2, \dots\}$  is the largest element with

$$f_{\mathcal{S}_k}(w_{k+1}; \xi_k) \leq f_{\mathcal{S}_k}(w_k; \xi_k) + \eta \alpha_k \nabla f_{\mathcal{S}_k}(w_k; \xi_k)^T s_k. \quad (6.7)$$

1. 7: **end for**

---

As previously mentioned, the (subsampled) Hessian-vector products required in Algorithm 6.1 can be computed efficiently in the context of many machine learning applications. For instance, one such case in the following.


---

## Page 6

**Example 6.2.** Consider a binary classification problem where the training function is given by the logistic loss with an  $\ell_2$ -norm regularization parameterized by  $\lambda > 0$ :

$$R_n(w) = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i w^T x_i)) + \frac{\lambda}{2} \|w\|^2. \quad (6.8)$$

A (subsampled) Hessian-vector product can be computed efficiently by observing that

$$\nabla^2 f_{\mathcal{S}_k^H}(w_k; \xi_k^H) d = \frac{1}{|\mathcal{S}_k^H|} \sum_{i \in \mathcal{S}_k^H} \frac{\exp(-y_i w^T x_i)}{(1 + \exp(-y_i w^T x_i))^2} (x_i^T d) x_i + \lambda d.$$

To quantify the step computation cost in an inexact Newton-CG framework such as Algorithm 6.1, let  $g_{cost}$  be the cost of computing a gradient estimate  $\nabla f_{\mathcal{S}_k}(w_k; \xi_k)$  and  $\text{factor} \times g_{cost}$  denote the cost of one Hessian-vector product. If the maximum number of CG iterations,  $\max_{cg}$ , is performed for every outer iteration, then the step computation cost in Algorithm 6.1 is

$$\max_{cg} \times \text{factor} \times g_{cost} + g_{cost}.$$

In a deterministic inexact Newton-CG method for minimizing the empirical risk  $R_n$ , i.e., when  $|\mathcal{S}_k^H| = |\mathcal{S}_k| = n$  for all  $k \in \mathbb{N}$ , the factor is at least 1 and  $\max_{cg}$  would typically be chosen as 5, 20, or more, leading to an iteration that is many times the cost of an SG iteration. However, in a stochastic framework using Hessian sub-sampling, the factor can be chosen to be sufficiently small such that  $\max_{cg} \times \text{factor} \approx 1$ , leading to a per-iteration cost proportional to that of SG.

Implicit in this discussion is the assumption that the gradient sample size  $|\mathcal{S}_k|$  is large enough so that taking subsamples for the Hessian estimate is sensible. If, by contrast, the algorithm were to operate in the stochastic regime of SG where  $|\mathcal{S}_k|$  is small and gradients are very noisy, then it may be necessary to choose  $|\mathcal{S}_k^H| > |\mathcal{S}_k|$  so that Hessian approximations do not corrupt the step. In such circumstances, the method would be far less attractive than SG. Therefore, the subsampled Hessian-free Newton method outlined here is only recommended when  $\mathcal{S}_k$  is large. This is why, in Figure 6.1, the Hessian-free Newton method is illustrated only with an arrow to the right, i.e., in the direction of larger sample sizes.

Convergence of Algorithm 6.1 is easy to establish when minimizing a strongly convex empirical risk measure  $F = R_n$  when  $\mathcal{S}_k \leftarrow \{1, \dots, n\}$  for all  $k \in \mathbb{N}$ , i.e., when full gradients are always used. In this case, a benefit of employing CG to solve (6.6) is that it immediately improves upon the direction employed in a steepest descent iteration. Specifically, when initialized at zero, it produces in its first iteration a scalar multiple of the steepest descent direction  $-\nabla F(w_k)$ , and further iterations monotonically improve upon this step (in terms of minimizing a quadratic model of the form in (6.3)) until the Newton step is obtained, which is done so in at most  $d$  iterations of CG (in exact arithmetic). Therefore, by using any number of CG iterations, convergence can be established using standard techniques to choose the stepsize  $\alpha_k$  [114]. When exact Hessians are also used, the rate of convergence can be controlled through the accuracy with which the systems (6.4b) are solved. Defining  $r_k := \nabla^2 F(w_k) s_k + \nabla F(w_k)$  for all  $k \in \mathbb{N}$ , the iteration can enjoy a linear, superlinear, or quadratic rate of convergence by controlling  $\|r_k\|_2$ , where for the superlinear rates one must have  $\{\|r_k\|_2 / \|\nabla F(w_k)\|_2\} \rightarrow 0$  [47].

When the Hessians are subsampled (i.e.,  $\mathcal{S}_k^H \subset \mathcal{S}_k$  for all  $k \in \mathbb{N}$ ), it has not been shown that the rate of convergence is faster than linear; nevertheless, the reduction in the number of iterations required to produce a good approximate solution is often significantly lower than if no Hessian information is used in the algorithm.


---

## Page 7

### 6.1.2 Dealing with Nonconvexity

Hessian-free Newton methods are routinely applied for the solution of nonconvex problems. In such cases, it is common to employ a *trust region* [37] instead of a line search and to introduce an additional condition in Step 5 of Algorithm 6.1: terminate CG if a candidate solution  $s_k$  is a direction of negative curvature, i.e.,  $s_k^T \nabla^2 f_{\mathcal{S}_k^H}(w_k; \xi_k^H) s_k < 0$  [149]. A number of more sophisticated strategies have been proposed throughout the years with some success, but none have proved to be totally satisfactory or universally accepted.

Instead of coping with indefiniteness, one can focus on strategies for ensuring positive (semi)definite Hessian approximations. One of the most attractive ways of doing this in the context of machine learning is to employ a (subsampling) Gauss-Newton approximation to the Hessian, which is a matrix of the form

$$G_{\mathcal{S}_k^H}(w_k; \xi_k^H) = \frac{1}{|\mathcal{S}_k^H|} \sum_{i \in \mathcal{S}_k^H} J_h(w_k, \xi_{k,i})^T H_\ell(w_k, \xi_{k,i}) J_h(w_k, \xi_{k,i}). \quad (6.9)$$

Here, the matrix  $J_h$  captures the stochastic gradient information for the prediction function  $h(x; w)$ , whereas the matrix  $H_\ell$  only captures the second order information for the (convex) loss function  $\ell(h, y)$ ; see §6.3 for a detailed explanation. As before, one can directly code the product of this matrix times a vector without forming the matrix components explicitly. This approach has been applied successfully in the training of deep neural networks [12, 100].

We mention in passing that there has been much discussion about the role that negative curvature and saddle points play in the optimization of deep neural networks; see e.g., [44, 70, 35]. Numerical tests designed to probe the geometry of the objective function in the neighborhood of a minimizer when training a deep neural network have shown the presence of negative curvature. It is believed that the inherent stochasticity of the SG method allows it to navigate efficiently through this complex landscape, but it is not known whether classical techniques to avoid approaching saddle points will prove to be successful for either batch or stochastic methods.

## 6.2 Stochastic Quasi-Newton Methods

One of the most important developments in the field of nonlinear optimization came with the advent of *quasi-Newton* methods. These methods construct approximations to the Hessian using only gradient information, and are applicable for convex and nonconvex problems. Versions that scale well with the number of variables, such as *limited memory* methods, have proved to be effective in a wide range of applications where the number of variables can be in the millions. It is therefore natural to ask whether quasi-Newton methods can be extended to the stochastic setting arising in machine learning. Before we embark on this discussion, let us review the basic principles underlying quasi-Newton methods, focusing on the most popular scheme, namely BFGS [27, 60, 68, 146].

In the spirit of Newton's method (6.4), the BFGS iteration for minimizing a twice continuously differentiable function  $F$  has the form

$$w_{k+1} \leftarrow w_k - \alpha_k H_k \nabla F(w_k), \quad (6.10)$$

where  $H_k$  is a symmetric positive definite approximation of  $(\nabla^2 F(w_k))^{-1}$ . This form of the iteration is consistent with (6.4), but the signifying feature of a quasi-Newton scheme is that the sequence


---

## Page 8

$\{H_k\}$  is updated dynamically by the algorithm rather than through a second-order derivative computation at each iterate. Specifically, in the BFGS method, the new inverse Hessian approximation is obtained by defining the iterate and gradient displacements

$$s_k := w_{k+1} - w_k \quad \text{and} \quad v_k := \nabla F(w_{k+1}) - \nabla F(w_k),$$

then setting

$$H_{k+1} \leftarrow \left( I - \frac{v_k s_k^T}{s_k^T v_k} \right)^T H_k \left( I - \frac{v_k s_k^T}{s_k^T v_k} \right) + \frac{s_k s_k^T}{s_k^T v_k}. \quad (6.11)$$

One important aspect of this update is that it ensures that the secant equation  $H_{k+1}^{-1} s_k = v_k$  holds, meaning that a second-order Taylor expansion is satisfied along the most recent displacement (though not necessarily along other directions).

A remarkable fact about the BFGS iteration (6.10)–(6.11) is that it enjoys a local superlinear rate of convergence [51], and this with only first-order information and without the need for any linear system solves (which are required by Newton’s method for it to be quadratically convergent). However, a number of issues need to be addressed to have an effective method in practice. For one thing, the update (6.11) yields dense matrices, even when the exact Hessians are sparse, restricting its use to small and midsize problems. A common solution for this is to employ a *limited memory* scheme, leading to a method such as L-BFGS [97, 113]. A key feature of this approach is that the matrices  $\{H_k\}$  need not be formed explicitly; instead, each product of the form  $H_k \nabla F(w_k)$  can be computed using a formula that only requires recent elements of the sequence of displacement pairs  $\{(s_k, v_k)\}$  that have been saved in storage. Such an approach incurs per-iteration costs of order  $\mathcal{O}(d)$ , and delivers practical performance that is significantly better than an full gradient iteration, though the rate of convergence is only provably linear.

### 6.2.1 Deterministic to Stochastic

Let us consider the extension of a quasi-Newton approach from the deterministic to the stochastic setting arising in machine learning. The iteration now takes the form

$$w_{k+1} \leftarrow w_k - \alpha_k H_k g(w_k, \xi_k). \quad (6.12)$$

Since we are interested in large-scale problems, we assume that (6.12) implements an L-BFGS scheme, which avoids the explicit construction of  $H_k$ . A number of questions arise when considering (6.12). We list them now with some proposed solutions.

**Theoretical Limitations** The convergence rate of a stochastic iteration such as (6.12) cannot be faster than sublinear [1]. Given that SG also has a sublinear rate of convergence, what benefit, if any, could come from incorporating  $H_k$  in (6.12)? This is an important question. As it happens, one can see a benefit of  $H_k$  in terms of the *constant* that appears in the sublinear rate. Recall that for the SG method (Algorithm 4.1), the constant depends on  $L/c$ , which in turn depends on the conditioning of  $\{\nabla^2 F(w_k)\}$ . This is typical of first-order methods. In contrast, one can show [25] that if the sequence of Hessian approximations in (6.12) satisfies  $\{H_k\} \rightarrow \nabla^2 F(w_*)^{-1}$ , then the constant is independent of the conditioning of the Hessian. Although constructing Hessian approximations with this property might not be viable in practice, this fact suggests that stochastic quasi-Newton methods could be better equipped to cope with ill-conditioning than SG.


---

## Page 9

**Additional Per-Iteration Costs** The SG iteration is very inexpensive, requiring only the evaluation of  $g(w_k, \xi_k)$ . The iteration (6.12), on the other hand, also requires the product  $H_k g(w_k, \xi_k)$ , which is known to require  $4md$  operations where  $m$  is the *memory* in the L-BFGS updating scheme. Assuming for concreteness that the cost of evaluating  $g(w_k, \xi_k)$  is exactly  $d$  operations (using only one sample) and that the memory parameter is set to the typical value of  $m = 5$ , one finds that the stochastic quasi-Newton method is 20 times more expensive than SG. Can the iteration (6.12) yield fast enough progress as to offset this additional per-iteration cost? To address this question, one need only observe that the calculation just mentioned focuses on the gradient  $g(w_k, \xi_k)$  being based on a single sample. However, when employing mini-batch gradient estimates, the additional cost of the iteration (6.12) is only marginal. (Mini-batches of size 256 are common in practice.) The use of mini-batches may therefore be considered essential when one contemplates the use of a stochastic quasi-Newton method. This mini-batch need not be large, as in the Hessian-free Newton method discussed in the previous section, but it should not be less than, say, 20 or 50 in light of the additional costs of computing the matrix-vector products.

**Conditioning of the Scaling Matrices** The BFGS formula (6.11) for updating  $H_k$  involves differences in gradient estimates computed in consecutive iterations. In stochastic settings, the gradients  $\{g(w_k, \xi_k)\}$  are noisy estimates of  $\{\nabla F(w_k)\}$ . This can cause the updating process to yield poor curvature estimates, which may have a detrimental rather than beneficial effect on the quality of the computed steps. Since BFGS, like all quasi-Newton schemes, is an *overwriting* process, the effects of even a single bad update may linger for numerous iterations. How could such effects be avoided in the stochastic regime? There have been various proposals to avoid differencing noisy gradient estimates. One possibility is to employ the same sample when computing gradient differences [18, 142]. An alternative approach that allows greater freedom in the choice of the stochastic gradient is to *decouple* the step computation (6.12) and the Hessian update. In this manner, one can employ a larger sample, if necessary, when computing the gradient displacement vector. We discuss these ideas further in §6.2.2.

It is worthwhile to note that if the gradient estimate  $g(w_k, \xi_k)$  does not have high variance, then standard BFGS updating can be applied without concerns. Therefore, in the rest of this section, we focus on algorithms that employ noisy gradient estimates in the step computation (6.12). This means, e.g., that we are not considering the potential to tie the method with noise reduction techniques described in §5, though such an idea is natural and could be effective in practice.

## 6.2.2 Algorithms

A straightforward adaptation of L-BFGS involves only the replacement of deterministic gradients with stochastic gradients throughout the iterative process. The displacement pairs might then be defined as

$$s_k := w_{k+1} - w_k \quad \text{and} \quad v_k := \nabla f_{S_k}(w_{k+1}, \xi_k) - \nabla f_{S_k}(w_k, \xi_k). \quad (6.13)$$

Note the use of the same seed  $\xi_k$  in the two gradient estimates, in order to address the issues related to noise mentioned above. If each  $f_i$  is strongly convex, then  $s_k^T v_k > 0$ , and positive definiteness of the updates is also maintained. Such an approach is sometimes referred to as *online L-BFGS* [142, 103].

One disadvantage of this method is the need to compute two, as opposed to only one, gradient estimate per iteration: one to compute the gradient displacement (namely,  $g(w_{k+1}, \xi_k)$ ) and another


---

## Page 10

(namely,  $g(w_{k+1}, \xi_{k+1})$ ) to compute the subsequent step. This is not too onerous, at least as long as the per-iteration improvement outweighs the extra per-iteration cost. A more worrisome feature is that updating the inverse Hessian approximations with *every* step may not be warranted, and may even be detrimental when the gradient displacement is based on a small sample, since it could easily represent a poor approximation of the action of the true Hessian of  $F$ .

An alternative strategy, which might better represent the action of the true Hessian even when  $g(w_k, \xi_k)$  has high variance, is to employ an alternative  $v_k$ . In particular, since  $\nabla F(w_{k+1}) - \nabla F(w_k) \approx \nabla^2 F(w_k)(w_{k+1} - w_k)$ , one could define

$$v_k := \nabla^2 f_{\mathcal{S}_k^H}(w_k; \xi_k^H) s_k, \quad (6.14)$$

where  $\nabla^2 f_{\mathcal{S}_k^H}(w_k; \xi_k^H)$  is a subsampled Hessian and  $|\mathcal{S}_k^H|$  is large enough to provide useful curvature information. As in the case of Hessian-free Newton from §6.1.1, the product (6.14) can be performed without explicitly constructing  $\nabla^2 f_{\mathcal{S}_k^H}(w_k; \xi_k^H)$ .

Regardless of the definition of  $v_k$ , when  $|\mathcal{S}_k^H|$  is much larger than  $|\mathcal{S}_k|$ , the cost of quasi-Newton updating is excessive due to the cost of computing  $v_k$ . To address this issue, the computation of  $v_k$  can be performed only after a sequence of iterations, so as to amortize costs. This leads to the idea of decoupling the step computation from the quasi-Newton update. This approach, which we refer to for convenience as SQN, performs a sequence of iterations of (6.12) with  $H_k$  fixed, then computes a new displacement pair  $(s_k, v_k)$  with  $s_k$  defined as in (6.13) and  $v_k$  set using one of the strategies outlined above. This pair replaces one of the old pairs in storage, which in turn defines the limited memory BFGS step.

To formalize all of these alternatives, we state the general stochastic quasi-Newton method presented as Algorithm 6.2, with some notation borrowed from Algorithm 6.1. In the method, the step computation is based on a collection of  $m$  displacement pairs  $\mathcal{P} = \{s_j, v_j\}$  in storage and the current stochastic gradient  $\nabla f_{\mathcal{S}_k}(w_k; \xi_k)$ , where the matrix-vector product in (6.12) can be computed through a *two-loop recursion* [113, 114]. To demonstrate the generality of the method, we note that the *online L-BFGS* method sets  $\mathcal{S}_k^H \leftarrow \mathcal{S}_k$  and `update pairs` = `true` in every iteration. In *SQN* using (6.14), on the other hand,  $|\mathcal{S}_k^H|$  should be chosen larger than  $|\mathcal{S}_k|$  and one sets `update pairs` = `true` only every, say, 10 or 20 iterations.

To guarantee that the BFGS update is well defined, each displacement pair  $(s_j, v_j)$  must satisfy  $s_j^T v_j > 0$ . In deterministic optimization, this issue is commonly addressed by either performing a line search (involving exact gradient computations) or modifying the displacement vectors (e.g., through *damping*) so that  $s_j^T v_j > 0$ , in which case one does ensure that (6.11) maintains positive definite approximations. However, these mechanisms have not been fully developed in the stochastic regime when exact gradient information is unavailable and the gradient displacement vectors are noisy. Simple ways to overcome these difficulties is to replace the Hessian matrix with a Gauss-Newton approximation or to introduce a combination of damping and regularization (say, through the addition of simple positive definite matrices).

There remains much to be explored in terms of stochastic quasi-Newton methods for machine learning applications. Experience has shown that some gains in performance can be achieved, but the full potential of the quasi-Newton schemes discussed above (and potentially others) is not yet known.


---

## Page 11

---

**Algorithm 6.2** Stochastic Quasi-Newton Framework

---

```
1: Choose an initial iterate  $w_1$  and initialize  $\mathcal{P} \leftarrow \emptyset$ .
2: Choose a constant  $m \in \mathbb{N}$ .
3: Choose a stepsize sequence  $\{\alpha_k\} \subset \mathbb{R}_{++}$ .
4: for  $k = 1, 2, \dots$ , do
5:   Generate realizations of  $\xi_k$  and  $\xi_k^H$  corresponding to  $\mathcal{S}_k$  and  $\mathcal{S}_k^H$ .
6:   Compute  $\hat{s}_k = H_k g(w_k, \xi_k)$  using the two-loop recursion based on the set  $\mathcal{P}$ .
7:   Set  $s_k \leftarrow -\alpha_k \hat{s}_k$ .
8:   Set  $w_{k+1} \leftarrow w_k + s_k$ .
9:   if update pairs then
10:    Compute  $s_k$  and  $v_k$  (based on the sample  $\mathcal{S}_k^H$ ).
11:    Add the new displacement pair  $(s_k, v_k)$  to  $\mathcal{P}$ .
12:    If  $|\mathcal{P}| > m$ , then remove eldest pair from  $\mathcal{P}$ .
13:  end if
14: end for
```

---

### 6.3 Gauss-Newton Methods

The Gauss-Newton method is a classical approach for nonlinear least squares, i.e., minimization problems in which the objective function is a sum of squares. This method readily applies for optimization problems arising in machine learning involving a least squares loss function, but the idea generalizes for other popular loss functions as well. The primary advantage of Gauss-Newton is that it constructs an approximation to the Hessian using only first-order information, and this approximation is guaranteed to be positive semidefinite, even when the full Hessian itself may be indefinite. The price to pay for this convenient representation is that it ignores second-order interactions between elements of the parameter vector  $w$ , which might mean a loss of curvature information that could be useful for the optimization process.

**Classical Gauss-Newton** Let us introduce the classical Gauss-Newton approach by considering a situation in which, for a given input-output pair  $(x, y)$ , the loss incurred by a parameter vector  $w$  is measured via a squared norm discrepancy between  $h(x; w) \in \mathbb{R}^d$  and  $y \in \mathbb{R}^d$ . Representing the input-output pair being chosen randomly via the subscript  $\xi$ , we may thus write

$$f(w; \xi) = \ell(h(x_\xi; w), y_\xi) = \frac{1}{2} \|h(x_\xi; w) - y_\xi\|_2^2.$$

Writing a second-order Taylor series model of this function in the vicinity of parameter vector  $w_k$  would involve its gradient and Hessian at  $w_k$ , and minimizing the resulting model (recall (6.3)) would lead to a Newton iteration. Alternatively, a Gauss-Newton approximation of the function is obtained by making an affine approximation of the prediction function inside the quadratic loss function. Letting  $J_h(\cdot; \xi)$  represent the Jacobian of  $h(x_\xi; \cdot)$  with respect to  $w$ , we have the approximation

$$h(x_\xi; w) \approx h(x_\xi; w_k) + J_h(w_k; \xi)(w - w_k),$$


---

## Page 12

which leads to

$$\begin{aligned}
f(w; \xi) &\approx \frac{1}{2} \|h(x_\xi; w_k) + J_h(w_k; \xi)(w - w_k) - y_\xi\|_2^2 \\
&= \frac{1}{2} \|h(x_\xi; w_k) - y_\xi\|_2^2 + (h(x_\xi; w_k) - y_\xi)^T J_h(w_k; \xi)(w - w_k) \\
&\quad + \frac{1}{2} (w - w_k)^T J_h(w_k; \xi)^T J_h(w_k; \xi)(w - w_k).
\end{aligned}$$

In fact, this approximation is similar to a second-order Taylor series model, except that the terms involving the second derivatives of the prediction function  $h$  with respect to the parameter vector have been dropped. The remaining second-order terms are those resulting from the positive curvature of the quadratic loss  $\ell$ . This leads to replacing the subsampled Hessian matrix (6.5) by the Gauss-Newton matrix

$$G_{\mathcal{S}_k^H}(w_k; \xi_k^H) = \frac{1}{|\mathcal{S}_k^H|} \sum_{i \in \mathcal{S}_k^H} J_h(w_k; \xi_{k,i})^T J_h(w_k; \xi_{k,i}). \quad (6.15)$$

Since the Gauss-Newton matrix only differs from the true Hessian by terms that involve the factors  $h(x_\xi; w_k) - y_\xi$ , these two matrices are the same when the loss is equal to zero, i.e., when  $h(x_\xi; w_k) = y_\xi$ .

A challenge in the application of a Gauss-Newton scheme is that the Gauss-Newton matrix is often singular or nearly singular. In practice, this is typically handled by regularizing it by adding to it a positive multiple of the identity matrix. For least-squares loss functions, the inexact Hessian-free Newton methods of §6.1 and the stochastic quasi-Newton methods of §6.2 with gradient displacement vectors defined as in (6.14) can be applied with (regularized) Gauss-Newton approximations. This has the benefit that the scaling matrices are guaranteed to be positive definite.

The computational cost of the Gauss-Newton method depends on the dimensionality of the prediction function. When the prediction function is scalar-valued, the Jacobian matrix  $J_h$  is a single row whose elements are already being computed as an intermediate step in the computation of the stochastic gradient  $\nabla f(w; \xi)$ . However, this is no longer true when the dimensionality is larger than one since then computing the stochastic gradient vector  $\nabla f(w; \xi)$  does not usually require the explicit computation of all rows of the Jacobian matrix. This happens, for instance, in deep neural networks when one uses back propagation [134, 135].

**Generalized Gauss-Newton** Gauss-Newton ideas can also be generalized for other standard loss functions [141]. To illustrate, let us consider a slightly more general situation in which loss is measured by a composition of an arbitrary convex loss function  $\ell(h, y)$  and a prediction function  $h(x; w)$ . Combining the affine approximation of the prediction function  $h(x_\xi; w)$  with a second order Taylor expansion of the loss function  $\ell$  leads to the generalized Gauss-Newton matrix

$$G_{\mathcal{S}_k^H}(w_k; \xi_k^H) = \frac{1}{|\mathcal{S}_k^H|} \sum_{i \in \mathcal{S}_k^H} J_h(w_k; \xi_{k,i})^T H_\ell(w_k; \xi_{k,i}) J_h(w_k; \xi_{k,i}) \quad (6.16)$$

(recall (6.9)), where  $H_\ell(w_k; \xi) = \frac{\partial^2 \ell}{\partial h^2}(h(x_\xi; w_k), y_\xi)$  captures the curvature of the loss function  $\ell$ . This can be seen as a generalization of (6.15) in which  $H_\ell = I$ .

When training a deep neural network, one may exploit this generalized strategy by redefining  $\ell$  and  $h$  so that as much as possible of the network's computation is formally performed by  $\ell$  rather than by  $h$ . If this can be done in such a way that convexity of  $\ell$  is maintained, then one can faithfully


---

## Page 13

capture second-order terms for  $\ell$  using the generalized Gauss-Newton scheme. Interestingly, in many useful situations, this strategy gives simpler and more elegant expressions for  $H_\ell$ . For instance, probability estimation problems often reduce to using logarithmic losses of the form  $f(w; \xi) = -\log(h(x_\xi; w))$ . The generalized Gauss-Newton matrix then reduces to

$$\begin{aligned} G_{\mathcal{S}_k^H}(w_k; \xi_k^H) &= \frac{1}{|\mathcal{S}_k^H|} \sum_{i \in \mathcal{S}_k^H} J_h(w_k; \xi_{k,i})^T \frac{1}{h(w; \xi_{k,i})^2} J_h(w_k; \xi_{k,i}) \\ &= \frac{1}{|\mathcal{S}_k^H|} \sum_{i \in \mathcal{S}_k^H} \nabla f(w; \xi_{k,i}) \nabla f(w; \xi_{k,i})^T, \end{aligned} \quad (6.17)$$

which does not require explicit computation of the Jacobian  $J_h$ .

## 6.4 Natural Gradient Method

We have seen that Newton's method is invariant to *linear* transformations of the parameter vector  $w$ . By contrast, the natural gradient method [5, 6] aims to be invariant with respect to all differentiable and invertible transformations. The essential idea consists of formulating the gradient descent algorithm in the space of prediction functions rather than specific parameters. Of course, the actual computation takes place with respect to the parameters, but accounts for the anisotropic relation between the parameters and the decision function. That is, in parameter space, the natural gradient algorithm will move the parameters more quickly along directions that have a small impact on the decision function, and more cautiously along directions that have a large impact on the decision function.

We remark at the outset that many authors [119, 99] propose *quasi-natural-gradient* methods that are strikingly similar to the *quasi-Newton* methods described in §6.2. The natural gradient approach therefore offers a different justification for these algorithms, one that involves qualitatively different approximations. It should also be noted that research on the design of methods inspired by the natural gradient is ongoing and may lead to markedly different algorithms [33, 76, 101].

**Information Geometry** In order to directly formulate the gradient descent in the space of prediction functions, we must elucidate the geometry of this space. Amari's work on information geometry [6] demonstrates this for parametric density estimation. The space  $\mathcal{H}$  of prediction functions for such a problem is a family of densities  $h_w(x)$  parametrized by  $w \in \mathcal{W}$  and satisfying the normalization condition

$$\int h_w(x) dx = 1 \quad \text{for all } w \in \mathcal{W}.$$

Assuming sufficient regularity, the derivatives of such densities satisfy the identity

$$\forall t > 0 \quad \int \frac{\partial^t h_w(x)}{\partial w^t} dx = \frac{\partial^t}{\partial w^t} \int h_w(x) dx = \frac{\partial^t 1}{\partial w^t} = 0. \quad (6.18)$$

To elucidate the geometry of the space  $\mathcal{H}$ , we seek to quantify how the density  $h_w$  changes when one adds a small quantity  $\delta w$  to its parameter. We achieve this in a statistically meaningful way by observing the Kullback-Leibler (KL) divergence

$$D_{KL}(h_w \| h_{w+\delta w}) = \mathbb{E}_{h_w} \left[ \log \left( \frac{h_w(x)}{h_{w+\delta w}(x)} \right) \right], \quad (6.19)$$


---

## Page 14

where  $\mathbb{E}_{h_w}$  denotes the expectation with respect to the distribution  $h_w$ . Note that (6.19) only depends on the values of the two density functions  $h_w$  and  $h_{w+\delta w}$  and therefore is invariant with respect to any invertible transformation of the parameter  $w$ . Approximating the divergence with a second-order Taylor expansion, one obtains

$$\begin{aligned} D_{KL}(h_w \| h_{w+\delta w}) &= \mathbb{E}_{h_w} [\log(h_w(x)) - \log(h_{w+\delta w}(x))] \\ &\approx -\delta w^T \mathbb{E}_{h_w} \left[ \frac{\partial \log(h_w(x))}{\partial w} \right] - \frac{1}{2} \delta w^T \mathbb{E}_{h_w} \left[ \frac{\partial^2 \log(h_w(x))}{\partial w^2} \right] \delta w, \end{aligned}$$

which, after observing that (6.18) implies that the first-order term is null, yields

$$D_{KL}(h_w \| h_{w+\delta w}) \approx \frac{1}{2} \delta w^T G(w) \delta w. \quad (6.20)$$

This is a quadratic form defined by the *Fisher information* matrix

$$G(w) := -\mathbb{E}_{h_w} \left[ \frac{\partial^2 \log(h_w(x))}{\partial w^2} \right] = -\mathbb{E}_{h_w} \left[ \left( \frac{\partial \log(h_w(x))}{\partial w} \right) \left( \frac{\partial \log(h_w(x))}{\partial w} \right)^T \right], \quad (6.21)$$

where the latter equality follows again from (6.18). The second form of  $G(w)$  is often preferred because it makes clear that the Fisher information matrix  $G(w)$  is symmetric and always positive semidefinite, though not necessarily positive definite.

The relation (6.20) means that the KL divergence behaves locally like a norm associated with  $G(w)$ . Therefore, every small region of  $\mathcal{H}$  looks like a small region of a Euclidean space. However, as we traverse larger regions of  $\mathcal{H}$ , we cannot ignore that the matrix  $G(w)$  changes. Such a construction defines a *Riemannian geometry*.<sup>7</sup>

Suppose, for instance, that we move along a smooth path connecting two densities, call them  $h_{w_0}$  and  $h_{w_1}$ . A parametric representation of the path can be given by a differentiable function, for which we define

$$\phi : t \in [0, 1] \mapsto \phi(t) \in \mathcal{W} \text{ with } \phi(0) = w_0 \text{ and } \phi(1) = w_1.$$

We can compute the length of the path by viewing it as a sequence of infinitesimal segments  $[\phi(t), \phi(t+dt)]$  whose length is given by (6.20), i.e., the total length is

$$D_\phi = \int_0^1 \sqrt{\left( \frac{d\phi}{dt}(t) \right)^T G(\phi(t)) \left( \frac{d\phi}{dt}(t) \right)} dt.$$

An important tool for the study of Riemannian geometries is the characterization of its *geodesics*, i.e., the shortest paths connecting two points. In a Euclidean space, the shortest path between two points is always the straight line segment connecting them. In a Riemannian space, on the other hand, the shortest path between two points can be curved and does not need to be unique. Such considerations are relevant to optimization since every iterative optimization algorithm can be viewed as attempting to follow a particular path connecting the initial point  $w_0$  to the optimum  $w_*$ . In particular, following the shortest path is attractive because it means that the algorithm reaches the optimum after making the fewest number of changes to the optimization variables, hopefully requiring the least amount of computation.

---

<sup>7</sup>The objective of information geometry [6] is to exploit the Riemannian structure of parametric families of density functions to gain geometrical insights on the fundamental statistical phenomena. The natural gradient algorithm is only a particular aspect of this broader goal [5].


---

## Page 15

**Natural Gradient** Let us now assume that the space  $\mathcal{H}$  of prediction functions  $\{h_w : w \in \mathcal{W}\}$  has a Riemannian geometry locally described by an identity of the form (6.20). We seek an algorithm that minimizes a functional  $F : h_w \in \mathcal{H} \mapsto F(h_w) = F(w) \in \mathbb{R}$  and is invariant with respect to differentiable invertible transformations of the parameters represented by the vector  $w$ .

Each iteration of a typical iterative optimization algorithm computes a new iterate  $h_{w_{k+1}}$  on the basis of information pertaining to the current iterate  $h_{w_k}$ . Since we can only expect this information to be valid in a small region surrounding  $h_{w_k}$ , we restrict our attention to algorithms that make a step from  $h_{w_k}$  to  $h_{w_{k+1}}$  of some small length  $\eta_k > 0$ . The number of iterations needed to reach the optimum then depends directly on the length of the path followed by the algorithm, which is desired to be as short as possible. Unfortunately, it is rarely possible to exactly follow a geodesic using only local information. We can, however, formulate the greedy strategy that

$$h_{w_{k+1}} = \arg \min_{h \in \mathcal{H}} F(h) \quad \text{s.t.} \quad D(h_{w_k} \| h) \leq \eta_k^2, \quad (6.22)$$

and use (6.20) to reformulate this problem in terms of the parameters:

$$w_{k+1} = \arg \min_{w \in \mathcal{W}} F(w) \quad \text{s.t.} \quad \frac{1}{2}(w - w_k)^T G(w_k) (w - w_k) \leq \eta_k^2. \quad (6.23)$$

The customary derivation of the natural gradient algorithm handles the constraint in (6.23) using a Lagrangian formulation with Lagrange multiplier  $1/\alpha_k$ . In addition, since  $\eta_k$  is assumed small, it replaces  $F(w)$  in (6.23) by the first-order approximation  $F(w_k) + \nabla F(w_k)^T (w - w_k)$ . These two choices lead to the expression

$$w_{k+1} = \arg \min_{w \in \mathcal{W}} \nabla F(w_k)^T (w - w_k) + \frac{1}{2\alpha_k} (w - w_k)^T G(w_k) (w - w_k),$$

the optimization of the right-hand side of which leads to the natural gradient iteration

$$w_{k+1} = w_k - \alpha_k G^{-1}(w_k) \nabla F(w_k). \quad (6.24)$$

We can also replace  $F(w)$  in (6.23) by a noisy first-order approximation, leading to a stochastic natural gradient iteration where  $\nabla F(w_k)$  in (6.24) is replaced by a stochastic gradient estimate.

Both batch and stochastic versions of (6.24) resemble the quasi-Newton update rules discussed in §6.2. Instead of multiplying the gradient by the inverse of an approximation of the Hessian (which is not necessarily positive definite), it employs the positive semidefinite matrix  $G(w_k)$  that expresses the local geometry of the space of prediction functions. In principle, this matrix does not even take into account the objective function  $F$ . However, as we shall now describe, one finds that these choices are all closely related in practice.

**Practical Natural Gradient** Because the discovery of the natural gradient algorithm is closely associated with information geometry, nearly all its applications involve density estimation [5, 33] or conditional probability estimation [119, 76, 99] using objective functions that are closely connected to the KL divergence. Natural gradient in this context is closely related to Fisher's scoring algorithm [118]. For instance, in the case of density estimation, the objective is usually the negative log likelihood

$$F(w) = \frac{1}{n} \sum_{i=1}^n -\log(h_w(x_i)) \approx \text{constant} + D_{KL}(P \| h_w),$$


---

## Page 16

where  $\{x_1, \dots, x_n\}$  represent independent training samples from an unknown distribution  $P$ . Recalling the expression of the Fisher information matrix (6.21) then clarifies its connection with the Hessian as one finds that

$$G(w) = -\mathbb{E}_{h_w} \left[ \frac{\partial^2 \log(h_w(x))}{\partial w^2} \right] \quad \text{and} \quad \nabla^2 F(w) = -\mathbb{E}_P \left[ \frac{\partial^2 \log(h_w(x))}{\partial w^2} \right].$$

These two expressions do not coincide in general because the expectations involve different distributions. However, when the natural gradient algorithm approaches the optimum, the parametric density  $h_{w_k}$  ideally approaches the true distribution  $P$ , in which case the Fisher information matrix  $G(w_k)$  approaches the Hessian matrix  $\nabla^2 F(w_k)$ . This means that the natural gradient algorithm and Newton's method perform very similarly as optimality is approached.

Although it is occasionally possible to determine a convenient analytic expression [33, 76], the numerical computation of the Fisher information matrix  $G(w_k)$  in large learning systems is generally very challenging. Moreover, estimating the expectation (6.21) with, say, a Monte-Carlo approach is usually prohibitive due to the cost of sampling the current density estimate  $h_{w_k}$ .

Several authors [119, 99] suggest to use instead a subset of training examples and compute a quantity of the form

$$\tilde{G}(w_k) = \frac{1}{|S_k|} \sum_{i \in S_k} \left( \frac{\partial \log(h_w(x_i))}{\partial w} \bigg|_{w_k} \right) \left( \frac{\partial \log(h_w(x_i))}{\partial w} \bigg|_{w_k} \right)^T.$$

Although such algorithms are essentially equivalent to the generalized Gauss-Newton schemes described in §6.3, the natural gradient perspective comes with an interesting insight into the relation between the generalized Gauss-Newton matrix (6.17) and the Hessian matrix (6.5). Similar to the equality (6.21), these two matrices would be equal if the expectation was taken with respect to the model distribution  $h_w$  instead of the empirical sample distribution.

## 6.5 Methods that Employ Diagonal Scalings

The methods that we have discussed so far in this section are forced to overcome the fact that when employing an iteration involving an  $\mathbb{R}^d \times \mathbb{R}^d$  scaling matrix, one needs to ensure that the improved per-iteration progress outweighs the added per-iteration cost. We have seen that these added costs can be as little as  $4md$  operations and therefore amount to a moderate multiplicative factor on the cost of each iteration.

A strategy to further reduce this multiplicative factor, while still incorporating second-order-type information, is to restrict attention to *diagonal* or *block-diagonal* scaling matrices. Rather than perform a more general linear transformation through a symmetric positive definite matrix (i.e., corresponding to a scaling and rotation of the direction), the incorporation of a diagonal scaling matrix only has the effect of scaling the individual search direction components. This can be efficiently achieved by multiplying each coefficients of the gradient vector by the corresponding diagonal term of the scaling matrix, or, when the prediction function is linear, by adaptively renormalizing the input pattern coefficients [133].

**Computing Diagonal Curvature** A first family of algorithms that we consider directly computes the diagonal terms of the Hessian or Gauss-Newton matrix, then divides each coefficient of


---

## Page 17

the stochastic gradient vector  $g(w_k, \xi_k)$  by the corresponding diagonal term. Since the computation overhead of this operation is very small, it becomes important to make sure that the estimation of the diagonal terms of the curvature matrix is very efficient.

For instance, in the context of deep neural networks, [12] describes a back-propagation algorithm to efficiently compute the diagonal terms of the squared Jacobian matrix  $J_h(w_k; \xi_k)^T J_h(w_k; \xi_k)$  that appears in the expression of the Gauss-Newton matrix (6.15). Each iteration of the proposed algorithm picks a training example, computes the stochastic gradient  $g(w_k, \xi_k)$ , updates a running estimate of the diagonal coefficients of the Gauss-Newton matrix by

$$[G_k]_i = (1 - \lambda)[G_{k-1}]_i + \lambda[J_h(w_k; \xi_k)^T J_h(w_k; \xi_k)]_{ii} \quad \text{for some } 0 < \lambda < 1,$$

then performs the scaled stochastic weight update

$$[w_{k+1}]_i = [w_k]_i - \left( \frac{\alpha}{[G_k]_i + \mu} \right) [g(w_k, \xi_k)]_i.$$

The small regularization constant  $\mu > 0$  is introduced to handle situations where the Gauss-Newton matrix is singular or nearly singular. Since the computation of the diagonal of the squared Jacobian has a cost that is comparable to the cost of the computation of the stochastic gradient, each iteration of this algorithm is roughly twice as expensive as a first-order stochastic gradient iteration. The experience described in [12] shows that improvement in per-iteration progress can be sufficient to modestly outperform a well-tuned SG algorithm.

After describing this algorithm in later work [89, §9.1], the authors make two comments that illustrate well how this algorithm was used in practice. They first observe that the curvature only changes very slowly in the particular type of neural network that was considered. Due to this observation, a natural idea is to further reduce the computational overhead of the method by estimating the ratios  $\alpha/([G_{k+1}]_i + \mu)$  only once every few epochs, for instance using a small subset of examples as in (6.15). The authors also mention that, as a rule of thumb, this diagonal scheme typically improves the convergence speed by only a factor of three relative to SG. Therefore, it might be more enlightening to view such an algorithm as a scheme to periodically retune a first-order SG approach rather than as a complete second-order method.

**Estimating Diagonal Curvature** Instead of explicitly computing the diagonal terms of the curvature matrix, one can follow the template of §6.2 and directly estimate the diagonal  $[H_k]_i$  of the inverse Hessian using displacement pairs  $\{(s_k, v_k)\}$  as defined in (6.13). For instance, [18] proposes to compute the scaling terms  $[H_k]_i$  with the running average

$$[H_{k+1}]_i = (1 - \lambda)[H_k]_i + \lambda \text{Proj} \left( \frac{[s_k]_i}{[v_k]_i} \right),$$

where  $\text{Proj}(\cdot)$  represents a projection onto a predefined positive interval. It was later found that a direct application of (6.13) after a parameter update introduces a correlated noise that ruins the curvature estimate [19, §3]. Moreover, correcting this problem made the algorithm perform substantially worse because the chaotic behavior of the rescaling factors  $[H_k]_i$  makes the choice of the stepsize  $\alpha$  very difficult.

These problems can be addressed with a combination of two ideas [19, §5]. The first idea consists of returning to estimating the diagonal of the Hessian instead of the diagonal of this inverse, which


---

## Page 18

amounts to working with the ratio  $[v_k]_i/[s_k]_i$  instead of  $[s_k]_i/[v_k]_i$ . The second idea ensures that the effective stepsizes are monotonically decreasing by replacing the running average by the sum

$$[G_{k+1}]_i = [G_k]_i + \text{Proj} \left( \frac{[v_k]_i}{[s_k]_i} \right).$$

This effectively constructs a separate diminishing stepsize sequence  $\alpha/[G_k]_i$  for each coefficient of the parameter vector. Keeping the curvature estimates in a fixed positive interval ensures that the effective stepsizes decrease at the rate  $\mathcal{O}(1/k)$  as prescribed by Theorem 4.7, while taking the local curvature into account. This combination was shown to perform very well when the input pattern coefficients have very different variances [19], something that often happens, e.g., in text classification problems.

**Diagonal Rescaling without Curvature** The algorithms described above often require some form of regularization to handle situations where the Hessian matrix is (nearly) singular. To illustrate why this is needed, consider, e.g., optimization of the convex objective function

$$F(w_1, w_2) = \frac{1}{2}w_1^2 + \log(e^{w_2} + e^{-w_2}),$$

for which ones finds

$$\nabla F(w_1, w_2) = \begin{bmatrix} w_1 \\ \tanh(w_2) \end{bmatrix} \quad \text{and} \quad \nabla^2 F(w_1, w_2) = \begin{bmatrix} 1 & 0 \\ 0 & 1/\cosh^2(w_2) \end{bmatrix}.$$

Performing a first-order gradient method update from a starting point of (3, 3) yields the negative gradient step  $-\nabla F \approx [-3, -1]$ , which unfortunately does not point towards the optimum, namely the origin. Moreover, rescaling the step with the inverse Hessian actually gives a worse update direction  $-(\nabla^2 F)^{-1} \nabla F \approx [-3, -101]$  whose large second component requires a small stepsize to keep the step well contained. Batch second-order optimization algorithms can avoid having to guess a good stepsize by using, e.g., line search techniques. Stochastic second-order algorithms, on the other hand, cannot rely on such procedures as easily.

This problem is of great concern in situations where the objective function is nonconvex. For instance, optimization algorithms for deep neural networks must navigate around saddle points and handle near-singular curvature matrices. It is therefore tempting to consider diagonal rescaling techniques that simply ensure equal progress along each axis, rather than attempt to approximate curvature very accurately.

For instance, RMSPROP [152] estimates the average magnitude of each element of the stochastic gradient vector  $g(w_k, \xi_k)$  by maintaining the running averages

$$[R_k]_i = (1 - \lambda)[R_{k-1}]_i + \lambda[g(w_k, \xi_k)]_i^2.$$

The rescaling operation then consists in dividing each component of  $g(w_k, \xi_k)$  by the square root of the corresponding running average, ensuring that the expected second moment of each coefficient of the rescaled gradient is close to the unity:

$$[w_{k+1}]_i = [w_k]_i - \frac{\alpha}{\sqrt{[R_k]_i + \mu}} [g(w_k, \xi_k)]_i.$$


---

## Page 19

This surprisingly simple approach has shown to be very effective for the optimization of deep neural networks. Various improvements have been proposed [162, 84] on an empirical basis. The theoretical explanation of this performance on nonconvex problems is still the object of active research [43].

The popular ADAGRAD algorithm [54] can be viewed as a member of this family that replaces the running average by a sum:

$$[R_k]_i = [R_{k-1}]_i + [g(w_k, \xi_k)]_i^2.$$

In this manner, the approach constructs a sequence of diminishing effective stepsizes  $\alpha/\sqrt{[R_k]_i + \mu}$  for each coefficient of the parameter vector. This algorithm was initially proposed and analyzed for the optimization of (not necessarily strongly) convex functions for which SG theory suggests diminishing stepsizes that scale with  $\mathcal{O}(1/\sqrt{k})$ . ADAGRAD is also known to perform well on deep learning networks, but one often finds that its stepsizes decrease too aggressively early in the optimization [162].

**Structural Methods** The performance of deep neural network training can of course be improved by employing better optimization algorithms. However, it can also be improved by changing the structure of the network in a manner that facilitates the optimization [80, 74]. We now describe one of these techniques, batch normalization [80], and discuss its relation to diagonal second-order methods.

Consider a particular fully connected layer in a deep neural network of the form discussed in §2.2. Using the notation of equation (2.4), the vector  $x_i^{(j)}$  represents the input values of layer  $j$  when the network is processing the  $i$ -th training example. Omitting the layer index for simplicity, let  $\hat{x}_i = (x_i^{(j)}, 1)$  denote the input vector augmented with an additional unit coefficient and let  $\hat{w}_r = (W_{r1}, \dots, W_{rd_{j-1}}, b_r)$  be the  $r$ th row of the matrix  $W_j$  augmented with the  $r$ th coefficient of the bias vector  $b_j$ . The layer outputs are then obtained by applying the activation function to the quantities  $s_r = \hat{w}_r^T \hat{x}_i$  for  $r \in \{1, \dots, d_j\}$ . Assuming for simplicity that all other parameters of the network are kept fixed, we can write

$$F(\hat{w}_1, \dots, \hat{w}_{d_j}) = \frac{1}{n} \sum_{i=1}^n \ell(h(\hat{w}_1^T \hat{x}_i, \hat{w}_2^T \hat{x}_i, \dots, \hat{w}_{d_j}^T \hat{x}_i), y_i),$$

where  $h(s_1, \dots, s_{d_j})$  encapsulates all subsequent layers in the network. The diagonal block of the Gauss-Newton matrix (6.16) corresponding to the parameters  $\hat{w}_r$  then has the form

$$G_{[r]} = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \left[ \left( \frac{dh}{ds_r} \right)^T \left( \frac{\partial^2 \ell}{\partial h^2} \right) \left( \frac{dh}{ds_r} \right) \right] \hat{x}_i \hat{x}_i^T, \quad (6.25)$$

which can be viewed as a weighted second moment matrix of the augmented input vectors  $\{\hat{x}_i\}_{i \in \mathcal{S}}$ . In particular, this matrix is perfectly conditioned if the weighted distribution of the layer inputs is white, i.e., they have zero mean and a unit covariance matrix. This could be achieved by first preprocessing the inputs by an affine transform that whitens their weighted distribution.

Two simplifications can drastically reduce the computational cost of this operation. First, we can ignore the bracketed coefficient in (6.25) and assume that we can use the same whitening transformation for all outputs  $r \in \{1, \dots, d_j\}$ . Second, we can ignore the input cross-correlations and simply ensure that each input variable has zero mean and unit variance by replacing the input


---

## Page 20

vector coefficients  $\hat{x}_i[t]$  for each  $t \in \{1, \dots, d_{j-1}\}$  by the linearly transformed values  $\alpha_t \hat{x}_i[t] + \beta_t$ . Despite these simplifications, this normalization operation is very likely to improve the second order properties of the objective function. An important detail here is the computation of the normalization constants  $\alpha_t$  and  $\beta_t$ . Estimating the mean and the standard deviation of each input with a simple running average works well if one expects these quantities to change very slowly. This is unfortunately not true in recent neural networks.<sup>8</sup>

Batch normalization [80] defines a special kind of neural network layer that performs this normalization using statistics collected with the current mini-batch of examples. The back-propagation algorithm that computes the gradients must of course be adjusted to account for the on-the-fly computation of the normalization coefficients. Assuming that one uses sufficiently large mini-batches, computing the statistics in this manner ensures that the normalization constants are very fresh. This comes at the price of making the output of the neural network on a particular training pattern dependent on the other patterns in the mini-batch. Since these other examples are *a priori* random, this amounts to generating additional noise in the stochastic gradient optimization. Although the variance of this noise is poorly controlled, inserting batch normalization layers in various points of a deep neural network is extremely effective and is now standard practice. Whether one can achieve the same improvement with more controlled techniques remains to be seen.

## 7 Other Popular Methods

Some optimization methods for machine learning are not well characterized as being within the two-dimensional schematic introduced in §3.4 (see Figure 3.3 on page 20), yet represent fundamentally unique approaches that offer theoretical and/or practical advantages. The purpose of this section is to discuss a few such ideas, namely, gradient methods with momentum, accelerated gradient methods, and coordinate descent methods. For ease of exposition, we introduce these techniques under the assumption that one is minimizing a continuously differentiable (not necessarily convex) function  $F : \mathbb{R}^n \rightarrow \mathbb{R}$  and that full gradients can be computed in each iteration. Then, after each technique is introduced, we discuss how they may be applied in stochastic settings.

### 7.1 Gradient Methods with Momentum

Gradient methods with momentum are procedures in which each step is chosen as a combination of the steepest descent direction and the most recent iterate displacement. Specifically, with an initial point  $w_1$ , scalar sequences  $\{\alpha_k\}$  and  $\{\beta_k\}$  that are either predetermined or set dynamically, and  $w_0 := w_1$ , these methods are characterized by the iteration

$$w_{k+1} \leftarrow w_k - \alpha_k \nabla F(w_k) + \beta_k (w_k - w_{k-1}). \quad (7.1)$$

Here, the latter is referred to as the momentum term, which, recursively, maintains the algorithm's movement along previous search directions.

The iteration (7.1) can be motivated in various ways; e.g., it is named after the fact that it represents a discretization of a certain second-order ordinary differential equation with friction. Of

---

<sup>8</sup>This used to be true in the 1990s because neural networks were using bounded activation functions such as the sigmoid  $s(x) = 1/(1 + e^{-x})$ . However, many recent results were achieved using the ReLU activation function  $s(x) = \max\{0, x\}$  which is unbounded and homogeneous. The statistics of the intermediate variables in such network can change extremely quickly during the first phases of the optimization process [86].


---

## Page 21

course, when  $\beta_k = 0$  for all  $k \in \mathbb{N}$ , it reduces to the steepest descent method. When  $\alpha_k = \alpha$  and  $\beta_k = \beta$  for some constants  $\alpha > 0$  and  $\beta > 0$  for all  $k \in \mathbb{N}$ , it is referred to as the heavy ball method [123], which is known to yield a superior rate of convergence as compared to steepest descent with a fixed stepsize for certain functions of interest. For example, when  $F$  is a strictly convex quadratic with minimum and maximum eigenvalues given by  $c > 0$  and  $L \geq c$ , respectively, steepest descent and the heavy ball method each yield a linear rate of convergence (in terms of the distance to the solution converging to zero) with contraction constants respectively given by

$$\frac{\kappa - 1}{\kappa + 1} \quad \text{and} \quad \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \quad \text{where} \quad \kappa := \frac{L}{c} \geq 1. \quad (7.2)$$

Choosing  $(\alpha, \beta)$  to achieve these rates requires knowledge of  $(c, L)$ , which might be unavailable. Still, even without this knowledge, the heavy ball method often outperforms steepest descent.

Additional connections with (7.1) can be made when  $F$  is a strictly convex quadratic. In particular, if  $(\alpha_k, \beta_k)$  is chosen optimally for all  $k \in \mathbb{N}$ , in the sense that the pair is chosen to solve

$$\min_{(\alpha, \beta)} F(w_k - \alpha \nabla F(w_k) + \beta(w_k - w_{k-1})), \quad (7.3)$$

then (7.1) is exactly the linear conjugate gradient (CG) algorithm. While the heavy ball method is a stationary iteration (in the sense that the pair  $(\alpha, \beta)$  is fixed), the CG algorithm is nonstationary and its convergence behavior is relatively more complex; in particular, the step-by-step behavior of CG depends on the eigenvalue distribution of the Hessian of  $F$  [69]. That said, in contrast to the heavy ball method, CG has a finite convergence guarantee. This, along with the fact that problems with favorable eigenvalue distributions are quite prevalent, has led to the great popularity of CG in a variety of situations. More generally, nonlinear CG methods, which also follow the procedure in (7.1), can be viewed as techniques that approximate the optimal values defined by (7.3) when  $F$  is not quadratic.

An alternative view of the heavy ball method is obtained by expanding (7.1):

$$w_{k+1} \leftarrow w_k - \alpha \sum_{j=1}^k \beta^{k-j} \nabla F(w_k);$$

thus, each step can be viewed as an exponentially decaying average of past gradients. By writing the iteration this way, one can see that the steps tend to accumulate contributions in directions of persistent descent, while directions that oscillate tend to get cancelled, or at least remain small.

This latter interpretation provides some intuitive explanation as to why a stochastic heavy ball method, and stochastic gradient methods with momentum in general, might be successful in various settings. In particular, their practical performance have made them popular in the community working on training deep neural networks [150]. Replacing the true gradient with a stochastic gradient in (7.1), one obtains an iteration that, over the long run, tends to continue moving in directions that the stochastic gradients suggest are ones of improvement, whereas movement is limited along directions along which contributions of many stochastic gradients cancel each other out. Theoretical guarantees about the inclusion of momentum in stochastic settings are elusive, and although practical gains have been reported [92, 150], more experimentation is needed.


---

## Page 22

## 7.2 Accelerated Gradient Methods

A method with an iteration similar to (7.1), but with its own unique properties, is the accelerated gradient method proposed by Nesterov [107]. Written as a two-step procedure, it involves the updates

$$\begin{aligned} \tilde{w}_k &\leftarrow w_k + \beta_k(w_k - w_{k-1}) \\ \text{and } w_{k+1} &\leftarrow \tilde{w}_k - \alpha_k \nabla F(\tilde{w}_k), \end{aligned} \tag{7.4}$$

which leads to the condensed form

$$w_{k+1} \leftarrow w_k - \alpha_k \nabla F(w_k + \beta_k(w_k - w_{k-1})) + \beta_k(w_k - w_{k-1}). \tag{7.5}$$

In this manner, it is easy to compare the approach with (7.1). In particular, one can describe their difference as being a reversal in the order of computation. In (7.1), one can imagine taking the steepest descent step and then applying the momentum term, whereas (7.5) results when one follows the momentum term first, then applies a steepest descent step (with the gradient evaluated at  $\tilde{w}_k$ , not at  $w_k$ ).

While this difference may appear to be minor, it is well known that (7.5) with appropriately chosen  $\alpha_k = \alpha > 0$  for all  $k \in \mathbb{N}$  and  $\{\beta_k\} \nearrow 1$  leads to an *optimal* iteration complexity when  $F$  is convex and continuously differentiable with a Lipschitz continuous gradient. Specifically, while in such cases a steepest descent method converges with a distance to the optimal value decaying with a rate  $\mathcal{O}(\frac{1}{k})$ , the iteration (7.5) converges with a rate  $\mathcal{O}(\frac{1}{k^2})$ , which is provably the best rate that can be achieved by a gradient method. Unfortunately, no intuitive explanation as to how Nesterov's method achieves this optimal rate has been widely accepted. Still, one cannot deny the analysis and the practical gains that the technique has offered.

Acceleration ideas have been applied in a variety of other contexts as well, including for the minimization of nonsmooth convex functions; see [96]. For now, we merely mention that when applied in stochastic settings—with stochastic gradients employed in place of full gradients—one can only hope that acceleration might improve the constants in the convergence rate offered in Theorem 4.7; i.e., the rate itself cannot be improved [1].

## 7.3 Coordinate Descent Methods

Coordinate descent (CD) methods are among the oldest in the optimization literature. As their name suggests, they operate by taking steps along coordinate directions: one attempts to minimize the objective with respect to a single variable while all others are kept fixed, then other variables are updated similarly in an iterative manner. Such a simple idea is easy to implement, so it is not surprising that CD methods have a long history in many settings. Their limitations have been documented and well understood for many years (more on these below), but one can argue that their advantages were not fully recognized until recent work in machine learning and statistics demonstrated their ability to take advantage of certain structures commonly arising in practice.

The CD method for minimizing  $F : \mathbb{R}^d \rightarrow \mathbb{R}$  is given by the iteration

$$w_{k+1} \leftarrow w_k - \alpha_k \nabla_{i_k} F(w_k) e_{i_k}, \quad \text{where } \nabla_{i_k} F(w_k) := \frac{\partial F}{\partial w^{i_k}}(w_k), \tag{7.6}$$

$w^{i_k}$  represents the  $i_k$ -th element of the parameter vector, and  $e_{i_k}$  represents the  $i_k$ -th coordinate vector for some  $i_k \in \{1, \dots, d\}$ . In other words, the solution estimates  $w_{k+1}$  and  $w_k$  differ only in their  $i_k$ -th element as a result of a move in the  $i_k$ -th coordinate from  $w_k$ .


---

## Page 23

Specific versions of the CD method are defined by the manner in which the sequences  $\{\alpha_k\}$  and  $\{i_k\}$  are chosen. In some applications, it is possible to choose  $\alpha_k$  as the global minimizer of  $F$  from  $w_k$  along the  $i_k$ -th coordinate direction. An important example of this, which has contributed to the recent revival of CD methods, occurs when the objective function has the form  $F(w) = q(w) + \|w\|_1$  when  $q$  is a convex quadratic. Here, the exact minimization along each coordinate is not only possible, but desirable as it promotes the generation of sparse iterates; see also §8. More often, an exact one-dimensional minimization of  $F$  is not practical, in which case one is typically satisfied with  $\alpha_k$  yielding a sufficient reduction in  $F$  from  $w_k$ . For example, so-called second-order CD methods compute  $\alpha_k$  as the minimizer of a quadratic model of  $F$  along the  $i_k$ -th coordinate direction.

Concerning the choice of  $i_k$ , one could select it in each iteration in at least three different ways: by cycling through  $\{1, \dots, d\}$ ; by cycling through a random reordering of these indices (with the indices reordered after each set of  $d$  steps); or simply by choosing an index randomly with replacement in each iteration. Randomized CD algorithms (represented by the latter two strategies) have superior theoretical properties than the cyclic method (represented by the first strategy) as they are less likely to choose an unfortunate series of coordinates; more on this below. However, it remains an open question whether such randomized algorithms are more effective in typical applications.

We mention in passing that it is also natural to consider a *block-coordinate descent* method in which a handful of elements are chosen in each iteration. This is particularly effective when the objective function is (partially) block separable, which occurs in matrix factorization problems and least squares and logistic regression when each sample only depends on a few features. Clearly, in such settings, there are great advantages of a block-coordinate descent approach. However, since their basic properties are similar to the case of using a single index in each iteration, we focus on the iteration (7.6).

**Convergence Properties** Contrary to what intuition might suggest, a CD method is not guaranteed to converge when applied to minimize any given continuously differentiable function. Powell [127] gives an example of a *nonconvex* continuously differentiable function of three variables for which a cyclic CD method with  $\alpha_k$  chosen by exact one-dimensional minimization cycles without converging to a solution, i.e., at any limit point the gradient of  $F$  is nonzero. Although one can argue that failures of this type are unlikely to occur in practice, particularly for a randomized CD method, they show the weakness of the myopic strategy in a CD method that considers only one variable at a time. This is in contrast with the full gradient method, which guarantees convergence to stationarity even when the objective is nonconvex.

On the other hand, if the objective  $F$  is strongly convex, the CD method will not fail and one can establish a linear rate of convergence. The analysis is very simple when using a constant stepsize and we present one such result to provide some insights into the tradeoffs that arise with a CD approach. Let us assume that  $\nabla F$  is coordinate-wise Lipschitz continuous in the sense that, for all  $w \in \mathbb{R}^d$ ,  $i \in \{1, \dots, d\}$ , and  $\Delta w^i \in \mathbb{R}$ , there exists a constant  $L_i > 0$  such that

$$|\nabla_i F(w + \Delta w^i e_i) - \nabla_i F(w)| \leq L_i |\Delta w^i|. \quad (7.7)$$

We then define the maximum coordinate-wise Lipschitz constant as

$$\hat{L} := \max_{i \in \{1, \dots, d\}} L_i.$$


---

## Page 24

Loosely speaking,  $\widehat{L}$  is a bound on the curvature of the function along all coordinates.

**Theorem 7.1.** *Suppose that the objective function  $F : \mathbb{R}^d \rightarrow \mathbb{R}$  is continuously differentiable, strongly convex with constant  $c > 0$ , and has a gradient that is coordinate-wise Lipschitz continuous with constants  $\{L_1, \dots, L_d\}$ . In addition, suppose that  $\alpha_k = 1/\widehat{L}$  and  $i_k$  is chosen independently and uniformly from  $\{1, \dots, d\}$  for all  $k \in \mathbb{N}$ . Then, for all  $k \in \mathbb{N}$ , the iteration (7.6) yields*

$$\mathbb{E}[F(w_{k+1})] - F_* \leq \left(1 - \frac{c}{d\widehat{L}}\right)^k (F(w_1) - F_*). \quad (7.8)$$

*Proof.* As Assumption 4.1 leads to (4.3), coordinate-wise Lipschitz continuity of  $\nabla F$  yields

$$F(w_{k+1}) \leq F(w_k) + \nabla_{i_k} F(w_k)(w_{k+1}^{i_k} - w_k^{i_k}) + \frac{1}{2}\widehat{L}(w_{k+1}^{i_k} - w_k^{i_k})^2.$$

Thus, with the stepsize chosen as  $\alpha_k = 1/\widehat{L}$ , it follows that

$$F(w_{k+1}) - F(w_k) \leq -\frac{1}{\widehat{L}}\nabla_{i_k} F(w_k)^2 + \frac{1}{2\widehat{L}}\nabla_{i_k} F(w_k)^2 = -\frac{1}{2\widehat{L}}\nabla_{i_k} F(w_k)^2.$$

Taking expectations with respect to the distribution of  $i_k$ , one obtains

$$\begin{aligned} \mathbb{E}_{i_k}[F(w_{k+1})] - F(w_k) &\leq -\frac{1}{2\widehat{L}}\mathbb{E}_{i_k}[\nabla_{i_k} F(w_k)^2] \\ &= -\frac{1}{2\widehat{L}}\left(\frac{1}{d}\sum_{i=1}^d \nabla_i F(w_k)^2\right) \\ &= -\frac{1}{2\widehat{L}d}\|\nabla F(w_k)\|_2^2. \end{aligned}$$

Subtracting  $F_*$ , taking total expectations, recalling (4.12), and applying the above inequality repeatedly over the first  $k \in \mathbb{N}$  iterations yields (7.8), as desired.  $\square$

A brief overview of the convergence properties of other CD methods under the assumptions of Theorem 7.1 and in other settings is also worthwhile. First, it is interesting to compare the result of Theorem 7.1 with a result obtained using the *deterministic* Gauss-Southwell rule, in which  $i_k$  is chosen in each iteration according to the largest (in absolute value) component of the gradient. Using this approach, one obtains a similar result in which  $c$  is replaced by  $\hat{c}$ , the strong convexity parameter as measured by the  $\ell_1$ -norm [116]. Since  $\frac{c}{n} \leq \hat{c} \leq c$ , the Gauss-Southwell rule can be up to  $n$  times faster than a randomized strategy, but in the worst case it is no better (and yet more expensive due to the need to compute the full gradient vector in each iteration). Alternative methods have also been proposed in which, in each iteration, the index  $i_k$  is chosen randomly with probabilities proportional to  $L_i$  or according to the largest ratio  $|\nabla_i F(w_k)|/\sqrt{L_i}$  [93, 110]. These strategies also lead to linear convergence rates with constants that are better in some cases.

**Favorable Problem Structures** Theorem 7.1 shows that a simple randomized CD method is linearly convergent with constant dependent on the parameter dimension  $d$ . At first glance, this appears to imply that such a method is less efficient than a standard full gradient method. However, in situations in which  $d$  coordinate updates can be performed at cost similar to the evaluation of one full gradient, the method is competitive with a full gradient method both theoretically and in practice. Classes of problems in this category include those in which the objective function is

$$F(w) = \frac{1}{n}\sum_{j=1}^n \tilde{F}_j(x_j^T w) + \sum_{i=1}^d \hat{F}_i(w^i), \quad (7.9)$$


---

## Page 25

where, for all  $j \in \{1, \dots, n\}$ , the function  $\tilde{F}_j$  is continuously differentiable and dependent on the *sparse* data vector  $x_j$ , and, for all  $i \in \{1, \dots, d\}$ , the function  $\hat{F}_i$  is a (nonsmooth) regularization function. Such a form arises in least squares and logistic regression; see also §8.

For example, consider an objective function of the form

$$f(w) = \frac{1}{2} \|Xw - y\|_2^2 + \sum_{i=1}^d \hat{F}_i(w^i) \quad \text{with } X = [x_1 \ \cdots \ x_n],$$

which might be the original function of interest or might represent a model of (7.9) in which the first term is approximated by a convex quadratic model. In this setting,

$$\nabla_{i_k} f(w_{k+1}) = x_{i_k}^T r_{k+1} + \hat{F}'_{i_k}(w_{k+1}^{i_k}) \quad \text{with } r_{k+1} := Aw_{k+1} - b,$$

where, with  $w_{k+1} = w_k + \beta_k e_{i_k}$ , one may observe that  $r_{k+1} = r_k + \beta_k x_{i_k}$ . That is, since the residuals  $\{r_k\}$  can be updated with cost proportional to the number of nonzeros in  $x_{i_k}$ , call it  $\text{nnz}(x_{i_k})$ , the overall cost of computing the search direction in iteration  $k+1$  is also  $\mathcal{O}(\text{nnz}(x_{i_k}))$ . On the other hand, an evaluation of the entire gradient requires a cost of  $\mathcal{O}(\sum_{i=1}^n \text{nnz}(x_i))$ .

Overall, there exist various types of objective functions for which minimization by a CD method (with exact gradient computations) can be effective. These include objectives that are (partially) block separable (which arise in dictionary learning and non-negative matrix factorization problems), have structures that allow for the efficient computation of individual gradient components, or are diagonally dominant in the sense that each step along a coordinate direction yields a reduction in the objective proportional to that which would have been obtained by a step along the steepest descent direction. Additional settings in which CD methods are effective are online problems where gradient information with respect to a group of variables becomes available in time, in which case it is natural to update these variables as soon as information is received.

**Stochastic Dual Coordinate Ascent** What about *stochastic* CD methods? As an initial thought, one might consider the replacement of  $\nabla_{i_k} F(w_k)$  in (7.6) with a stochastic approximation, but this is not typical since one can usually as easily compute a  $d$ -dimensional stochastic gradient to apply an SG method. However, an interesting setting for the application of stochastic CD methods arises when one considers approaches to minimize a convex objective function of the form (7.9) by maximizing its *dual*. In particular, defining the convex conjugate of  $\tilde{F}_j$  as  $\tilde{F}_j^*(u) := \max_w (w^T u - \tilde{F}_j(w))$ , the Fenchel-Rockafellar dual of (7.9) when  $\hat{F}_i(\cdot) = \frac{\lambda}{2}(\cdot)^2$  for all  $i \in \{1, \dots, d\}$  is given by

$$F_{\text{dual}}(v) = \frac{1}{n} \sum_{j=1}^n -\tilde{F}_j^*(-v_j) - \frac{\lambda}{2} \left\| \frac{1}{\lambda n} \sum_{j=1}^n v_j x_j \right\|_2^2.$$

The stochastic dual coordinate ascent (SDCA) method [145] applied to a function of this form has an iteration similar to (7.6), except that negative gradient steps are replaced by gradient steps due to the fact that one aims to maximize the dual. At the conclusion of a run of the algorithm, the corresponding *primal* solution can be obtained as  $w \leftarrow \frac{1}{\lambda n} \sum_{j=1}^n v_j x_j$ . The per-iteration cost of this approach is on par with that of a stochastic gradient method.


---

## Page 26

**Parallel CD Methods** We close this section by noting that CD methods are also attractive when one considers the exploitation of parallel computation. For example, consider a multicore system in which the parameter vector  $w$  is stored in shared memory. Each core can then execute a CD iteration independently and in an asynchronous manner, where if  $d$  is large compared to the number of cores, then it is unlikely that two cores are attempting to update the same variable at the same time. Since, during the time it takes a core to perform an update, the parameter vector  $w$  has likely changed (due to updates produced by other cores), each update is being made based on slightly stale information. However, convergence of the method can be proved, and improves when one can bound the degree of staleness of each update. For further information and insight into these ideas, we refer the reader to [16, 98].

## 8 Methods for Regularized Models

Our discussion of structural risk minimization (see §2.3) highlighted the key role played by regularization functions in the formulation of optimization problems for machine learning. The optimization methods that we presented and analyzed in the subsequent sections (§3–§7) are all applicable when the objective function involves a smooth regularizer, such as the squared  $\ell_2$ -norm. In this section, we expand our investigation by considering optimization methods that handle the regularization as a distinct entity, in particular when that function is *nonsmooth*. One such regularizer that deserves special attention is the  $\ell_1$ -norm, which induces sparsity in the optimal solution vector. For machine learning, sparsity can be beneficial since it leads to simpler models, and hence can be seen as a form of *feature selection*, i.e., for biasing the optimization toward solutions where only a few elements of the parameter vector are nonzero.

Broadly, this section focuses on the nonsmooth optimization problem

$$\min_{w \in \mathbb{R}^d} \Phi(w) := F(w) + \lambda \Omega(w), \quad (8.1)$$

where  $F : \mathbb{R}^d \rightarrow \mathbb{R}$  includes the composition of a loss and prediction function,  $\lambda > 0$  is a regularization parameter, and  $\Omega : \mathbb{R}^d \rightarrow \mathbb{R}$  is a convex, nonsmooth regularization function. Specifically, we pay special attention to methods for solving the problem

$$\min_{w \in \mathbb{R}^d} \phi(w) := F(w) + \lambda \|w\|_1. \quad (8.2)$$

As discussed in §2, it is often necessary to solve a series of such problems over a sequence of values for the parameter  $\lambda$ . For further details in terms of problem (8.2), we refer the reader to [147, 9] for examples in a variety of applications. However, in our presentation of optimization methods, we assume that  $\lambda$  has been prescribed and is fixed. We remark in passing that (8.2) has as a special case the well-known LASSO problem [151] when  $F(w) = \|Aw - b\|_2^2$  for some  $A \in \mathbb{R}^{n \times d}$  and  $b \in \mathbb{R}^n$ .

Although nondifferentiable, the regularized  $\ell_1$  problem (8.2) has a structure that can be exploited in the design of algorithms. The algorithms that have been proposed can be grouped into classes of first- or second-order methods, and distinguished as those that either minimize the nonsmooth objective directly, as in a *proximal gradient* method, or by approximately minimizing a sequence of more complicated models, such as in a *proximal Newton* method.

There exist other *sparsity-inducing* regularizers besides the  $\ell_1$ -norm, including group-sparsity-inducing regularizers that combine  $\ell_1$ - and  $\ell_2$ -norms taken with respect to groups of variables


---

## Page 27

[147, 9], as well as the nuclear norm for optimization over spaces of matrices [32]. While we do not discuss other such regularizers in detail, our presentation of methods for  $\ell_1$ -norm regularized problems represents how methods for alternative regularizers can be designed and characterized.

As in the previous sections, we introduce the algorithms in this section under the assumption that  $F$  is continuously differentiable and full, batch gradients can be computed for it in each iteration, commenting on stochastic method variants once motivating ideas have been described.

## 8.1 First-Order Methods for Generic Convex Regularizers

The fundamental algorithm in unconstrained smooth optimization is the gradient method. For solving problem (8.1), the *proximal gradient* method represents a similar fundamental approach. Given an iterate  $w_k$ , a generic proximal gradient iteration, with  $\alpha_k > 0$ , is given by

$$w_{k+1} \leftarrow \arg \min_{w \in \mathbb{R}^d} \left( F(w_k) + \nabla F(w_k)^T (w - w_k) + \frac{1}{2\alpha_k} \|w - w_k\|_2^2 + \lambda \Omega(w) \right). \quad (8.3)$$

The term *proximal* refers to the presence of the third term in the minimization problem on the right-hand side, which encourages the new iterate to be close to  $w_k$ . Notice that if the regularization (i.e., last) term were not present, then (8.3) exactly recovers the gradient method update  $w_{k+1} \leftarrow w_k - \alpha_k \nabla F(w_k)$ ; hence, as previously, we refer to  $\alpha_k$  as the stepsize parameter. On the other hand, if the regularization term is present, then, similar to the gradient method, each new iterate is found by minimizing a model formed by a first-order Taylor series expansion of the objective function plus a simple scaled quadratic. Overall, the only thing that distinguishes a proximal gradient method from the gradient method is the regularization term, which is left untouched and included explicitly in each step computation.

To show how an analysis similar to those seen in previous sections can be used to analyze (8.3), we prove the following theorem. In it, we show that if  $F$  is strongly convex and its gradient function is Lipschitz continuous, then the iteration yields a global linear rate of convergence to the optimal objective value provided that the stepsizes are sufficiently small.

**Theorem 8.1.** *Suppose that  $F : \mathbb{R}^d \rightarrow \mathbb{R}$  is continuously differentiable, strongly convex with constant  $c > 0$ , and has a gradient that is Lipschitz continuous with constant  $L > 0$ . In addition, suppose that  $\alpha_k = \alpha \in (0, 1/L]$  for all  $k \in \mathbb{N}$ . Then, for all  $k \in \mathbb{N}$ , the iteration (8.3) yields*

$$\Phi(w_{k+1}) - \Phi(w_*) \leq (1 - \alpha c)^k (\Phi(w_1) - \Phi(w_*)),$$

where  $w_* \in \mathbb{R}^d$  is the unique global minimizer of  $\Phi$  in (8.1).

*Proof.* Since  $\alpha_k = \alpha \in (0, 1/L]$ , it follows from (4.3) that

$$\begin{aligned} \Phi(w_{k+1}) &= F(w_{k+1}) + \lambda \Omega(w_{k+1}) \\ &\leq F(w_k) + \nabla F(w_k)^T (w_{k+1} - w_k) + \frac{1}{2} L \|w_{k+1} - w_k\|_2^2 + \lambda \Omega(w_{k+1}) \\ &\leq F(w_k) + \nabla F(w_k)^T (w_{k+1} - w_k) + \frac{1}{2\alpha} \|w_{k+1} - w_k\|_2^2 + \lambda \Omega(w_{k+1}) \\ &\leq F(w_k) + \nabla F(w_k)^T (w - w_k) + \frac{1}{2\alpha} \|w - w_k\|_2^2 + \lambda \Omega(w) \quad \text{for all } w \in \mathbb{R}^d, \end{aligned}$$


---

## Page 28

where the last inequality follows since  $w_{k+1}$  is defined by (8.3). Representing  $w = w_k + d$ , we obtain

$$\begin{aligned}\Phi(w_{k+1}) &\leq F(w_k) + \nabla F(w_k)^T d + \frac{1}{2\alpha} \|d\|_2^2 + \lambda \Omega(w_k + d) \\ &\leq F(w_k) + \nabla F(w_k)^T d + \frac{1}{2} c \|d\|_2^2 - \frac{1}{2} c \|d\|_2^2 + \frac{1}{2\alpha} \|d\|_2^2 + \lambda \Omega(w_k + d) \\ &\leq F(w_k + d) + \lambda \Omega(w_k + d) - \frac{1}{2} c \|d\|_2^2 + \frac{1}{2\alpha} \|d\|_2^2 \\ &= \Phi(w_k + d) + \frac{1}{2} \left( \frac{1}{\alpha} - c \right) \|d\|_2^2,\end{aligned}$$

which for  $d = -\alpha c(w_k - w_*)$  means that

$$\begin{aligned}\Phi(w_{k+1}) &\leq \Phi(w_k - \alpha c(w_k - w_*)) + \frac{1}{2} \left( \frac{1}{\alpha} - c \right) \|\alpha c(w_k - w_*)\|_2^2 \\ &= \Phi(w_k - \alpha c(w_k - w_*)) + \frac{1}{2} \alpha c^2 (1 - \alpha c) \|w_k - w_*\|_2^2.\end{aligned}\tag{8.4}$$

On the other hand, since the  $c$ -strongly convex function  $\Phi$  satisfies (e.g., see [108, pp. 63–64])

$$\begin{aligned}\Phi(\tau w + (1 - \tau)\bar{w}) &\leq \tau \Phi(w) + (1 - \tau) \Phi(\bar{w}) - \frac{1}{2} c \tau (1 - \tau) \|w - \bar{w}\|_2^2 \\ &\text{for all } (w, \bar{w}, \tau) \in \mathbb{R}^d \times \mathbb{R}^d \times [0, 1],\end{aligned}\tag{8.5}$$

we have (considering  $\bar{w} = w_k$ ,  $w = w_*$ , and  $\tau = \alpha c \in (0, 1]$  in (8.5)) that

$$\begin{aligned}\Phi(w_k - \alpha c(w_k - w_*)) &\leq \alpha c \Phi(w_*) + (1 - \alpha c) \Phi(w_k) - \frac{1}{2} c (\alpha c) (1 - \alpha c) \|w_k - w_*\|_2^2 \\ &= \alpha c \Phi(w_*) + (1 - \alpha c) \Phi(w_k) - \frac{1}{2} \alpha c^2 (1 - \alpha c) \|w_k - w_*\|_2^2.\end{aligned}\tag{8.6}$$

Combining (8.6) with (8.4) and subtracting  $\Phi(w_*)$ , it follows that

$$\Phi(w_{k+1}) - \Phi(w_*) \leq (1 - \alpha c) (\Phi(w_k) - \Phi(w_*)).$$

The result follows by applying this inequality repeated over the first  $k \in \mathbb{N}$  iterations.  $\square$

One finds in Theorem 8.1 an identical result as for a gradient method for minimizing a smooth strongly convex function. As in such methods, the choice of the stepsize  $\alpha$  is critical in practice; the convergence guarantees demand that it be sufficiently small, but a value that is too small might unduly slow the optimization process.

The proximal gradient iteration (8.3) is practical only when the proximal mapping

$$\text{prox}_{\lambda \Omega, \alpha_k}(\tilde{w}) := \arg \min_{w \in \mathbb{R}^n} \left( \lambda \Omega(w) + \frac{1}{2\alpha_k} \|w - \tilde{w}\|_2^2 \right)$$

can be computed efficiently. This can be seen in the fact that the iteration (8.3) can equivalently be written as  $w_{k+1} \leftarrow \text{prox}_{\lambda \Omega, \alpha_k}(w_k - \alpha_k \nabla F(w_k))$ , i.e., the iteration is equivalent to applying a proximal mapping to the result of a gradient descent step. Situations when the proximal mapping is inexpensive to compute include when  $\Omega$  is the indicator function for a simple set, such as a polyhedral set, when it is the  $\ell_1$ -norm, or, more generally, when it is separable.

A stochastic version of the proximal gradient method can be obtained, not surprisingly, by replacing  $\nabla F(w_k)$  in (8.3) by a stochastic approximation  $g(w_k, \xi_k)$ . The iteration remains cheap to perform (since  $F(w_k)$  can be ignored as it does not effect the computed step). The resulting method attains similar behavior as a stochastic gradient method; analyses can be found, e.g., in [140, 8].

We now turn our attention to the most popular nonsmooth regularizer, namely the one defined by the  $\ell_1$  norm.


---

## Page 29

### 8.1.1 Iterative Soft-Thresholding Algorithm (ISTA)

In the context of solving the  $\ell_1$ -norm regularized problem (8.2), the proximal gradient method is

$$w_{k+1} \leftarrow \arg \min_{w \in \mathbb{R}^d} \left( F(w_k) + \nabla F(w_k)^T (w - w_k) + \frac{1}{2\alpha_k} \|w - w_k\|_2^2 + \lambda \|w\|_1 \right). \quad (8.7)$$

The optimization problem on the right-hand side of this expression is separable and can be solved in closed form. The solution can be written component-wise as

$$[w_{k+1}]_i \leftarrow \begin{cases} [w_k - \alpha_k \nabla F(w_k)]_i + \alpha_k \lambda & \text{if } [w_k - \alpha_k \nabla F(w_k)]_i < -\alpha_k \lambda \\ 0 & \text{if } [w_k - \alpha_k \nabla F(w_k)]_i \in [-\alpha_k \lambda, \alpha_k \lambda] \\ [w_k - \alpha_k \nabla F(w_k)]_i - \alpha_k \lambda & \text{if } [w_k - \alpha_k \nabla F(w_k)]_i > \alpha_k \lambda. \end{cases} \quad (8.8)$$

One also finds that this iteration can be written, with  $(\cdot)_+ := \max\{\cdot, 0\}$ , as

$$w_{k+1} \leftarrow \mathcal{T}_{\alpha_k \lambda}(w_k - \alpha_k \nabla F(w_k)), \quad \text{where } [\mathcal{T}_{\alpha_k \lambda}(\tilde{w})]_i = (|\tilde{w}_i| - \alpha_k \lambda)_+ \text{sgn}(\tilde{w}_i). \quad (8.9)$$

In this form,  $\mathcal{T}_{\alpha_k \lambda}$  is referred to as the soft-thresholding operator, which leads to the name *iterative soft-thresholding algorithm* (ISTA) being used for (8.7)–(8.8) [53, 42].

It is clear from (8.8) that the ISTA iteration induces sparsity in the iterates. If the steepest descent step with respect to  $F$  yields a component with absolute value less than  $\alpha_k \lambda$ , then that component is set to zero in the subsequent iterate; otherwise, the operator still has the effect of shrinking components of the solution estimates in terms of their magnitudes. When only a stochastic estimate  $g(w_k, \xi_k)$  of the gradient is available, it can be used instead of  $\nabla F(w_k)$ .

A variant of ISTA with acceleration (recall §7.2), known as FISTA [10], is popular in practice. We also mention that effective techniques have been developed for computing the stepsize  $\alpha_k$ , in ISTA or FISTA, based on an estimate of the Lipschitz constant of  $\nabla F$  or on curvature measured in recent iterations [10, 159, 11].

### 8.1.2 Bound-Constrained Methods for $\ell_1$ -norm Regularized Problems

By observing the structure created by the  $\ell_1$ -norm, one finds that an equivalent *smooth* reformulation of problem (8.2) is easily derived. In particular, by writing  $w = u - v$  where  $u$  plays the *positive part* of  $w$  while  $v$  plays the *negative part*, problem (8.2) can equivalently be written as

$$\min_{(u,v) \in \mathbb{R}^d \times \mathbb{R}^d} \tilde{\phi}(u, v) \quad \text{s.t. } (u, v) \geq 0, \quad \text{where } \tilde{\phi}(u, v) = F(u - v) + \lambda \sum_{i=1}^d (u_i + v_i). \quad (8.10)$$

Now with a bound-constrained problem in hand, one has at their disposal a variety of optimization methods that have been developed in the optimization literature.

The fundamental iteration for solving bound-constrained optimization problems is the *gradient projection* method. In the context of (8.10), the iteration reduces to

$$\begin{bmatrix} u_{k+1} \\ v_{k+1} \end{bmatrix} \leftarrow P_+ \left( \begin{bmatrix} u_k \\ v_k \end{bmatrix} - \alpha_k \begin{bmatrix} \nabla_u \tilde{\phi}(u_k, v_k) \\ \nabla_v \tilde{\phi}(u_k, v_k) \end{bmatrix} \right) = P_+ \left( \begin{bmatrix} u_k - \alpha_k \nabla F(u_k - v_k) - \alpha_k \lambda e \\ v_k + \alpha_k \nabla F(u_k - v_k) - \alpha_k \lambda e \end{bmatrix} \right), \quad (8.11)$$

where  $P_+$  projects onto the nonnegative orthant and  $e \in \mathbb{R}^d$  is a vector of ones.


---

## Page 30

Interestingly, the gradient projection method can also be derived from the perspective of a proximal gradient method where the regularization term  $\Omega$  is chosen to be the indicator function for the feasible set (a box). In this case, the mapping  $\mathcal{T}_{\alpha_k \lambda}$  is replaced by the projection operator onto the bound constraints, causing the corresponding proximal gradient method to coincide with the gradient projection method. In the light of this observation, one should expect the iteration (8.11) to inherit the property of being globally linearly convergent when  $F$  satisfies the assumptions of Theorem 8.1. However, since the variables in (8.10) have been split into positive and negative parts, this property is maintained *only if* the iteration maintains complementarity of each iterate pair, i.e., if  $[u_k]_i [v_k]_i = 0$  for all  $k \in \mathbb{N}$  and  $i \in \{1, \dots, d\}$ . This behavior is also critical for the practical performance of the method in general since, without it, the algorithm would not generate sparse solutions. In particular, maintaining this property allows the algorithm to be implemented in such a way that one effectively only needs  $d$  optimization variables.

A natural question that arises is whether the iteration (8.11) actually differs from an ISTA iteration, especially given that both are built upon proximal gradient ideas. In fact, the iterations can lead to the same update, but do not always. Consider, for example, an iterate  $w_k = u_k - v_k$  such that for  $i \in \{1, \dots, d\}$  one finds  $[w_k]_i > 0$  with  $[u_k]_i = [w_k]_i$  and  $[v_k]_i = 0$ . (A similar look, with various signs reversed, can be taken when  $[w_k]_i < 0$ .) If  $[w_k - \alpha_k \nabla F(w_k)]_i > \alpha_k \lambda$ , then (8.8) and (8.11) yield

$$\begin{aligned} [w_{k+1}]_i &\leftarrow [w_k - \alpha_k \nabla F(w_k)]_i - \alpha_k \lambda > 0 \\ \text{and } [u_{k+1}]_i &\leftarrow [u_k - \alpha_k \nabla F(w_k)]_i - \alpha_k \lambda > 0. \end{aligned}$$

However, it is important to note the step taken in the negative part; in particular, if  $[\nabla F(w_k)]_i \leq \lambda$ , then  $[v_{k+1}]_i \leftarrow 0$ , but, if  $[\nabla F(w_k)]_i > \lambda$ , then  $[v_{k+1}]_i \leftarrow \alpha_k \nabla F(w_k) - \alpha_k \lambda$ , in which case the lack of complementarity between  $u_{k+1}$  and  $v_{k+1}$  should be rectified. A more significant difference arises when, e.g.,  $[w_k - \alpha_k \nabla F(w_k)]_i < -\alpha_k \lambda$ , in which case (8.8) and (8.11) yield

$$\begin{aligned} [w_{k+1}]_i &\leftarrow [w_k - \alpha_k \nabla F(w_k)]_i + \alpha_k \lambda < 0, \\ [u_{k+1}]_i &\leftarrow 0, \\ \text{and } [v_{k+1}]_i &\leftarrow [v_k + \alpha_k \nabla F(w_k)]_i - \alpha_k \lambda > 0. \end{aligned}$$

The pair  $([u_{k+1}]_i, [v_{k+1}]_i)$  are complementary, but  $[w_{k+1}]_i$  and  $[-v_{k+1}]_i$  differ by  $[w_k]_i > 0$ .

Several first-order [59, 58] and second-order [138] gradient projection methods have been proposed to solve (8.2). Such algorithms should be preferred over similar techniques for general bound-constrained optimization, e.g., those in [163, 95], since such general techniques may be less effective by not exploiting the structure of the reformulation (8.10) of (8.2).

A stochastic projected gradient method, with  $\nabla F(w_k)$  replaced by  $g(w_k, \xi_k)$ , has similar convergence properties as a standard SG method; e.g., see [105]. These properties apply in the present context, but also apply when a proximal gradient method is used to solve (8.1) when  $\Omega$  represents the indicator function of a box constraint.

## 8.2 Second-Order Methods

We now turn our attention to methods that, like Newton's method for smooth optimization, are designed to solve regularized problems through successive minimization of second-order models constructed along the iterate sequence  $\{w_k\}$ . As in a proximal gradient method, the smooth function  $F$  is approximated by a Taylor series and the regularization term is kept unchanged. We focus on two classes of methods for solving (8.2): *proximal Newton* and *orthant-based* methods.


---

## Page 31

Both classes of methods fall under the category of *active-set* methods. One could also consider the application of an *interior-point* method to solve the bound-constrained problem (8.10) [114]. This, by its nature, constitutes a second-order method that would employ Hessians of  $F$  or corresponding quasi-Newton approximations. However, a disadvantage of the interior-point approach is that, by staying away from the boundary of the feasible region, it does not promote the fast generation of sparse solutions, which is in stark contrast with the methods described below.

### 8.2.1 Proximal Newton Methods

We use the term *proximal Newton* to refer to techniques that directly minimize the nonsmooth function arising as the sum of a quadratic model of  $F$  and the regularizer. In particular, for solving problem (8.2), a proximal Newton method is one that constructs, at each  $k \in \mathbb{N}$ , a model

$$q_k(w) = F(w_k) + \nabla F(w_k)^T (w - w_k) + \frac{1}{2}(w - w_k)^T H_k(w - w_k) + \lambda \|w\|_1 \approx \phi(w). \quad (8.12)$$

where  $H_k$  represents  $\nabla^2 F(w_k)$  or a quasi-Newton approximation of it. This model has a similar form as the one in (8.7), except that the simple quadratic is replaced by the quadratic form defined by  $H_k$ . A proximal Newton method would involve (approximately) minimizing this model to compute a trial iterate  $\tilde{w}_k$ , then a stepsize  $\alpha_k > 0$  would be taken from a predetermined sequence or chosen by a line search to ensure that the new iterate  $w_{k+1} \leftarrow w_k + \alpha_k(\tilde{w}_k - w_k)$  yields  $\Phi(w_{k+1}) < \Phi(w_k)$ .

Proximal Newton methods are more challenging to design, analyze, and implement than proximal gradient methods. That being said, they can perform better in practice once a few key challenges are addressed. The three ingredients below have proved to be essential in ensuring the practical success and scalability of a proximal Newton method. For simplicity, we assume throughout that  $H_k$  has been chosen to be positive definite.

**Choice of Subproblem Solver** The model  $q_k$  inherits the nonsmooth structure of  $\phi$ , which has the benefit of allowing a proximal Newton method to cross manifolds of nondifferentiability while simultaneously promoting sparsity of the iterates. However, the method needs to overcome the fact that the model  $q_k$  is nonsmooth, which makes the subproblem for minimizing  $q_k$  challenging. Fortunately, with the particular structure created by a quadratic plus an  $\ell_1$ -norm term, various methods are available for minimizing such a nonsmooth function. For example, coordinate descent is particularly well-suited in this context [153, 78] since the global minimizer of  $q_k$  along a coordinate descent direction can be computed analytically. Such a minimizer often occurs at a point of nondifferentiability (namely, when a component is zero), thus ensuring that the method will generate sparse iterates. Updating the gradient of the model  $q_k$  after each coordinate descent step can also be performed efficiently, even if  $H_k$  is given as a limited memory quasi-Newton approximation [137]. Numerical experiments have shown that employing a coordinate descent iteration is more efficient in certain applications than employing, say, an ISTA iteration to minimize  $q_k$ , though the latter is also a viable strategy in some applications.

**Inaccurate Subproblem Solves** A proximal Newton method needs to overcome the fact that, in large-scale settings, it is impractical to minimize  $q_k$  accurately for all  $k \in \mathbb{N}$ . Hence, it is natural to consider the computation of an inexact minimizer of  $q_k$ . The issue then becomes: what are practical, yet theoretically sufficient termination criteria when computing an approximate minimizer of the nonsmooth function  $q_k$ ? A common suggestion in the literature has been to use the norm of the


---

## Page 32

minimum-norm subgradient of  $q_k$  at a given approximate minimizer. However, this measure is not continuous, making it inadequate for these purposes.<sup>9</sup> Interestingly, the norm of an ISTA step is an appropriate measure. In particular, letting  $\text{ista}_k(w)$  represent the result of an ISTA step applied to  $q_k$  from  $w$ , the value  $\|\text{ista}_k(w) - w\|_2$  satisfies the following two key properties: (i) it equals zero if and only if  $w$  is a minimizer of  $q_k$ , and (ii) it varies continuously over  $\mathbb{R}^d$ .

Complete and sufficient termination criteria are then as follows: a trial point  $\tilde{w}_k$  represents a sufficiently accurate minimizer of  $q_k$  if, for some  $\eta \in [0, 1)$ , one finds

$$\|\text{ista}_k(\tilde{w}_k) - \tilde{w}_k\|_2 \leq \eta \|\text{ista}(w_k) - w_k\|_2 \quad \text{and} \quad q_k(\tilde{w}_k) < q_k(w_k).$$

The latter condition, requiring a decrease in  $q_k$ , must also be imposed since the ISTA criterion alone does not exert sufficient control to ensure convergence. Employing such criteria, it has been observed to be efficient to perform the minimization of  $q_k$  inaccurately at the start, and to increase the accuracy of the model minimizers as one approaches the solution. A superlinear rate of convergence for the overall proximal Newton iteration can be obtained by replacing  $\eta$  by  $\eta_k$  where  $\{\eta_k\} \searrow 0$ , along with imposing a stronger descent condition on the decrease in  $q_k$  [31].

**Elimination of Variables** Due to the structure created by the  $\ell_1$ -norm regularizer, it can be effective in some applications to first identify a set of *active* variables—i.e., ones that are predicted to be equal to zero at a minimizer for  $q_k$ —then compute an approximate minimizer of  $q_k$  over the remaining *free* variables. Specifically, supposing that a set  $\mathcal{A}_k \subseteq \{1, \dots, d\}$  of active variables has been identified, one may compute an (approximate) minimizer of  $q_k$  by (approximately) solving the reduced-space problem

$$\min_{w \in \mathbb{R}^d} q_k(w) \quad \text{s.t.} \quad [w]_i = 0, \quad i \in \mathcal{A}_k. \quad (8.13)$$

Moreover, during the minimization process for this problem, one may have reason to believe that the process may be improved by adding or removing elements from the active set estimate  $\mathcal{A}_k$ . In any case, performing the elimination of variables imposed in (8.13) has the effect of reducing the size of the subproblem, and can often lead to fewer iterations being required in the overall proximal Newton method. How should the active set  $\mathcal{A}_k$  be defined? A technique that has become popular recently is to use sensitivity information as discussed in more detail in the next subsection.

### 8.2.2 Orthant-Based Methods

Our second class of second-order methods is based on the observation that the  $\ell_1$ -norm regularized objective  $\phi$  in problem (8.2) is smooth in any orthant in  $\mathbb{R}^d$ . Based on this observation, orthant-based methods construct, in every iteration, a smooth quadratic model of the objective, then produce a search direction by minimizing this model. After performing a line search designed to reduce the objective function, a new orthant is selected and the process is repeated. This approach can be motivated by the success of second-order gradient projection methods for bound constrained optimization, which at every iteration employ a gradient projection search to identify an active set and perform a minimization of the objective function over a space of free variables to compute a search direction.

---

<sup>9</sup>Consider the one-dimensional case of having  $q_k(w) = |w|$ . The minimum-norm subgradient of  $q_k$  has a magnitude of 1 at all points, except at the minimizer  $w^* = 0$ ; hence, this norm does not provide a measure of proximity to  $w^*$ .


---

## Page 33

The selection of an orthant is typically done using sensitivity information. Specifically, with the minimum norm subgradient of  $\phi$  at  $w \in \mathbb{R}^d$ , which is given component-wise for all  $i \in \{1, \dots, d\}$  by

$$\hat{g}_i(w) = \begin{cases} [\nabla F(w)]_i + \lambda & \text{if } w_i > 0 \text{ or } \{w_i = 0 \text{ and } [\nabla F(w)]_i + \lambda < 0\} \\ [\nabla F(w)]_i - \lambda & \text{if } w_i < 0 \text{ or } \{w_i = 0 \text{ and } [\nabla F(w)]_i - \lambda > 0\} \\ 0 & \text{otherwise,} \end{cases} \quad (8.14)$$

the active orthant for an iterate  $w_k$  is characterized by the sign vector

$$\zeta_{k,i} = \begin{cases} \text{sgn}([w_k]_i) & \text{if } [w_k]_i \neq 0 \\ \text{sgn}(-[\hat{g}(w_k)]_i) & \text{if } [w_k]_i = 0. \end{cases} \quad (8.15)$$

Along these lines, let us also define the subsets of  $\{1, \dots, d\}$  given by

$$\mathcal{A}_k = \{i : [w_k]_i = 0 \text{ and } |[\nabla F(w_k)]_i| \leq \lambda\} \quad (8.16)$$

$$\text{and } \mathcal{F}_k = \{i : [w_k]_i \neq 0\} \cup \{i : [w_k]_i = 0 \text{ and } |[\nabla F(w_k)]_i| > \lambda\}, \quad (8.17)$$

where  $\mathcal{A}_k$  represents the indices of variables that are active and kept at zero while  $\mathcal{F}_k$  represents those that are free to move.

Given these quantities, an orthant-based method proceeds as follows. First, one computes the (approximate) solution  $d_k$  of the (smooth) quadratic problem

$$\begin{aligned} \min_{d \in \mathbb{R}^n} \quad & \hat{g}(w_k)^T d + \frac{1}{2} d^T H_k d \\ \text{s.t.} \quad & d_i = 0, \quad i \in \mathcal{A}_k, \end{aligned}$$

where  $H_k$  represents  $\nabla^2 F(x^k)$  or an approximation of it. Then, the algorithm performs a line search—over a path contained in the current orthant—to compute the next iterate. For example, one option is a projected backtracking line search along  $d_k$ , which involves computing the largest  $\alpha_k$  in a decreasing geometric sequence so

$$F(P_k(w_k + \alpha_k d_k)) < F(w_k).$$

Here,  $P_k(w)$  projects  $w \in \mathbb{R}^d$  onto the orthant defined by  $\zeta^k$ , i.e.,

$$[P_k(w)]_i = \begin{cases} w_i & \text{if } \text{sgn}(w_i) = \zeta_{k,i} \\ 0 & \text{otherwise.} \end{cases} \quad (8.18)$$

In this way, the initial and final points of an iteration lie in the same orthant. Orthant-based methods have proved to be quite useful in practice; e.g., see [7, 30].

**Commentary** Proximal Newton and orthant-based methods represent two efficient classes of second-order active-set methods for solving the  $\ell_1$ -norm regularized problem (8.2). The proximal Newton method is reminiscent of the sequential quadratic programming method (SQP) for constrained optimization; they both solve a complex subproblem that yields a useful estimate of the optimal active set. Although solving the piecewise quadratic model (8.12) is very expensive in general, the coordinate descent method has proven to be well suited for this task, and allows the


---

## Page 34

proximal Newton method to be applied to very large problems [79]. Orthant-based methods have shown to be equally useful, but in a more heuristic way, since some popular implementations lack convergence guarantees [58, 30]. Stochastic variants of both proximal Newton and orthant-based schemes can be devised in natural ways, and generally inherit the properties of stochastic proximal gradient methods as long as the Hessian approximations are forced to possess eigenvalues within a positive interval.

## 9 Summary and Perspectives

Mathematical optimization is one of the foundations of machine learning, touching almost every aspect of the discipline. In particular, numerical optimization algorithms, the main subject of this paper, have played an integral role in the transformational progress that machine learning has experienced over the past two decades. In our study, we highlight the dominant role played by the stochastic gradient method (SG) of Robbins and Monro [130], whose success derives from its superior work complexity guarantees. A concise, yet broadly applicable convergence and complexity theory for SG is presented here, providing insight into how these guarantees have translated into practical gains.

Although the title of this paper suggests that our treatment of optimization methods for machine learning is comprehensive, much more could be said about this rapidly evolving field. Perhaps most importantly, we have neither discussed nor analyzed at length the opportunities offered by parallel and distributed computing, which may alter our perspectives in the years to come. In fact, it has already been widely acknowledged that SG, despite its other benefits, may not be the best suited method for emerging computer architectures.

This leads to our discussion of a spectrum of methods that have the potential to surpass SG in the next generation of optimization methods for machine learning. These methods, such as those built on noise reduction and second-order techniques, offer the ability to attain improved convergence rates, overcome the adverse effects of high nonlinearity and ill-conditioning, and exploit parallelism and distributed architectures in new ways. There are important methods that are not included in our presentation—such as the alternating direction method of multipliers (ADMM) [57, 64, 67] and the expectation-maximization (EM) method and its variants [48, 160]—but our study covers many of the core algorithmic frameworks in optimization for machine learning, with emphasis on methods and theoretical guarantees that have the largest impact on practical performance.

With the great strides that have been made and the various avenues for continued contributions, numerical optimization promises to continue to have a profound impact on the rapidly growing field of machine learning.

## A Convexity and Analyses of SG

Our analyses of SG in §4 can be characterized as relying primarily on smoothness in the sense of Assumption 4.1. This has advantages and disadvantages. On the positive side, it allows us to prove convergence results that apply equally for the minimization of convex and nonconvex functions, the latter of which has been rising in importance in machine learning; recall §2.2. It also allows us to present results that apply equally to situations in which the stochastic vectors are unbiased estimators of the gradient of the objective, or when such estimators are scaled by a symmetric


---

## Page 35

positive definite matrix; recall (4.2). A downside, however, is that it requires us to handle the minimization of nonsmooth models separately, which we do in §8.

As an alternative, a common tactic employed by many authors is to leverage convexity instead of smoothness, allowing for the establishment of guarantees that can be applied in smooth and nonsmooth settings. For example, a typical approach for analyzing stochastic gradient-based methods is to commence with the following fundamental equations related to squared distances to the optimum:

$$\begin{aligned} \|w_{k+1} - w_*\|_2^2 - \|w_k - w_*\|_2^2 &= 2(w_{k+1} - w_k)^T(w_k - w_*) + \|w_{k+1} - w_k\|_2^2 \\ &= -2\alpha_k g(w_k, \xi_k)^T(w_k - w_*) + \alpha_k^2 \|g(w_k, \xi_k)\|_2^2. \end{aligned} \quad (\text{A.1})$$

Assuming that  $\mathbb{E}_{\xi_k}[g(w_k, \xi_k)] = \hat{g}(w_k) \in \partial F(w_k)$ , one then obtains

$$\begin{aligned} \mathbb{E}_{\xi_k}[\|w_{k+1} - w_*\|_2^2] - \|w_k - w_*\|_2^2 \\ = -2\alpha_k \hat{g}(w_k)^T(w_k - w_*) + \alpha_k^2 \mathbb{E}_{\xi_k}[\|g(w_k, \xi_k)\|_2^2], \end{aligned} \quad (\text{A.2})$$

which has certain similarities with (4.10a). One can now introduce an assumption of convexity to bound the first term on the right-hand side in this expression; in particular, convexity offers the subgradient inequality

$$\hat{g}(w_k)^T(w_k - w_*) \geq F(w_k) - F(w_*) \geq 0$$

while strong convexity offers the stronger condition (4.11). Combined with a suitable assumption on the second moment of the stochastic subgradients to bound the second term in the expression, the entire right-hand side can be adequately controlled through judicious stepsize choices. The resulting analysis then has many similarities with that presented in §4, especially if one introduces an assumption about Lipschitz continuity of the gradients of  $F$  in order to translate results on decreases in the distance to the solution in terms of decreases in  $F$  itself. The interested reader will find a clear exposition of such results in [105].

Note, however, that one can see in (A.2) that analyses based on distances to the solution do not carry over easily to nonconvex settings or when (quasi-)Newton type steps are employed. In such situations, without explicit knowledge of  $w_*$ , one cannot easily ensure that the first term on the right-hand side can be bounded appropriately.

## B Proofs

*Inequality (4.3).* Under Assumption 4.1, one obtains

$$\begin{aligned} F(w) &= F(\bar{w}) + \int_0^1 \frac{\partial F(\bar{w} + t(w - \bar{w}))}{\partial t} dt \\ &= F(\bar{w}) + \int_0^1 \nabla F(\bar{w} + t(w - \bar{w}))^T(w - \bar{w}) dt \\ &= F(\bar{w}) + \nabla F(\bar{w})^T(w - \bar{w}) + \int_0^1 [\nabla F(\bar{w} + t(w - \bar{w})) - \nabla F(\bar{w})]^T(w - \bar{w}) dt \\ &\leq F(\bar{w}) + \nabla F(\bar{w})^T(w - \bar{w}) + \int_0^1 L\|t(w - \bar{w})\|_2\|w - \bar{w}\|_2 dt, \end{aligned}$$

from which the desired result follows.  $\square$


---

## Page 36

*Inequality (4.12).* Given  $w \in \mathbb{R}^d$ , the quadratic model

$$q(\bar{w}) := F(w) + \nabla F(w)^T (\bar{w} - w) + \frac{1}{2}c\|\bar{w} - w\|_2^2$$

has the unique minimizer  $\bar{w}_* := w - \frac{1}{c}\nabla F(w)$  with  $q(\bar{w}_*) = F(w) - \frac{1}{2c}\|\nabla F(w)\|_2^2$ . Hence, the inequality (4.11) with  $\bar{w} = w_*$  and any  $w \in \mathbb{R}^d$  yields

$$F_* \geq F(w) + \nabla F(w)^T (w_* - w) + \frac{1}{2}c\|w_* - w\|_2^2 \geq F(w) - \frac{1}{2c}\|\nabla F(w)\|_2^2,$$

from which the desired result follows.  $\square$

*Corollary 4.12.* Define  $G(w) := \|\nabla F(w)\|_2^2$  and let  $L_G$  be the Lipschitz constant of  $\nabla G(w) = 2\nabla^2 F(w)\nabla F(w)$ . Then,

$$\begin{aligned} G(w_{k+1}) - G(w_k) &\leq \nabla G(w_k)^T (w_{k+1} - w_k) + \frac{1}{2}L_G\|w_{k+1} - w_k\|_2^2 \\ &\leq -\alpha_k \nabla G(w_k)^T g(w_k, \xi_k) + \frac{1}{2}\alpha_k^2 L_G \|g(w_k, \xi_k)\|_2^2. \end{aligned}$$

Taking the expectation with respect to the distribution of  $\xi_k$ , one obtains from Assumptions 4.1 and 4.3 and the inequality (4.9) that

$$\begin{aligned} &\mathbb{E}_{\xi_k} [G(w_{k+1}) - G(w_k)] \\ &\leq -2\alpha_k \nabla F(w_k)^T \nabla^2 F(w_k)^T E_{\xi_k} [g(w_k, \xi_k)] + \frac{1}{2}\alpha_k^2 L_G E_{\xi_k} [\|g(w_k, \xi_k)\|_2^2] \\ &\leq 2\alpha_k \|\nabla F(w_k)\|_2 \|\nabla^2 F(w_k)\|_2 \|E_{\xi_k} [g(w_k, \xi_k)]\|_2 + \frac{1}{2}\alpha_k^2 L_G E_{\xi_k} [\|g(w_k, \xi_k)\|_2^2] \\ &\leq 2\alpha_k L \mu_G \|\nabla F(w_k)\|_2^2 + \frac{1}{2}\alpha_k^2 L_G (M + M_V \|\nabla F(w_k)\|_2^2). \end{aligned}$$

Taking the total expectation simply yields

$$\begin{aligned} &\mathbb{E}[G(w_{k+1})] - \mathbb{E}[G(w_k)] \\ &\leq 2\alpha_k L \mu_G \mathbb{E}[\|\nabla F(w_k)\|_2^2] + \frac{1}{2}\alpha_k^2 L_G (M + M_V \mathbb{E}[\|\nabla F(w_k)\|_2^2]). \end{aligned} \tag{B.1}$$

Recall that Theorem 4.10 establishes that the first component of this bound is the term of a convergent sum. The second component of this bound is also the term of a convergent sum since  $\sum_{k=1}^{\infty} \alpha_k^2$  converges and since  $\alpha_k^2 \leq \alpha_k$  for sufficiently large  $k \in \mathbb{N}$ , meaning that again the result of Theorem 4.10 can be applied. Therefore, the right-hand side of (B.1) is the term of a convergent sum. Let us now define

$$\begin{aligned} S_K^+ &= \sum_{k=1}^K \max(0, \mathbb{E}[G(w_{k+1})] - \mathbb{E}[G(w_k)]) \\ \text{and } S_K^- &= \sum_{k=1}^K \max(0, \mathbb{E}[G(w_k)] - \mathbb{E}[G(w_{k+1})]). \end{aligned}$$

Since the bound (B.1) is positive and forms a convergent sum, the nondecreasing sequence  $S_K^+$  is upper bounded and therefore converges. Since, for any  $K \in \mathbb{N}$ , one has  $G(w_K) = G(w_0) + S_K^+ - S_K^- \geq 0$ , the nondecreasing sequence  $S_K^-$  is upper bounded and therefore also converges. Therefore  $G(w_K)$  converges and Theorem 4.9 means that this limit must be zero.  $\square$


---

## Page 37

## References

- [1] A. Agarwal, P. L. Bartlett, P. Ravikumar, and M. J. Wainwright. Information-Theoretic Lower Bounds on the Oracle Complexity of Stochastic Convex Optimization. *IEEE Transactions on Information Theory*, 58(5):3235–3249, 2012.
- [2] Naman Agarwal, Brian Bullins, and Elad Hazan. Second order stochastic optimization in linear time. *arXiv preprint arXiv:1602.03943*, 2016.
- [3] Zeyuan Allen Zhu and Elad Hazan. Optimal black-box reductions between optimization objectives. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, *Advances in Neural Information Processing Systems 29*, pages 1606–1614. 2016.
- [4] S.-I. Amari. A theory of adaptive pattern classifiers. *IEEE Transactions on Electronic Computers*, EC-16:299–307, 1967.
- [5] S.-I. Amari. Natural gradient works efficiently in learning. *Neural Computation*, 10(2):251–276, 1998.
- [6] S.-I. Amari and H. Nagaoka. *Methods of Information Geometry*. American Mathematical Society, Providence, RI, USA, 1997.
- [7] Galen Andrew and Jianfeng Gao. Scalable training of  $l_1$ -regularized log-linear models. In *Proceedings of the 24th international conference on Machine learning*, pages 33–40. ACM, 2007.
- [8] Yves F Atchade, Gersende Fort, and Eric Moulines. On stochastic proximal gradient algorithms. *arXiv preprint arXiv:1402.2365*, 2014.
- [9] F. Bach, R. Jenatton, J. Mairal, and G. Obozinski. Optimization with sparsity-inducing penalties. *Foundations and Trends® in Machine Learning*, 4(1):1–106, 2012.
- [10] A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1):183–202, 2009.
- [11] S. Becker, E. J. Candès, and M. Grant. Templates for convex cone problems with applications to sparse signal recovery. *Mathematical Programming Computation*, 3(3):165–218, 2011.
- [12] S. Becker and Y. LeCun. Improving the convergence of back-propagation learning with second-order methods. In D. Touretzky, G. Hinton, and T. Sejnowski, editors, *Proc. of the 1988 Connectionist Models Summer School*, pages 29–37, San Mateo, 1989. Morgan Kaufman.
- [13] D. P. Bertsekas. *Nonlinear Programming*. Athena Scientific, Belmont, Massachusetts, 1995.
- [14] D. P. Bertsekas. Incremental least squares methods and the extended Kalman filter. *SIAM Journal on Optimization*, 6(3):807–822, 1996.
- [15] D. P. Bertsekas. *Convex Optimization Algorithms*. Athena Scientific, Nashua, NH, USA, 2015.
- [16] D. P. Bertsekas and J. N. Tsitsiklis. *Parallel and Distributed Computation: Numerical Methods*. Prentice-Hall, Englewood Cliffs, NJ, USA, 1989.


---

## Page 38

- [17] D. Blatt, A. O. Hero, and H. Gauchman. A convergent incremental gradient method with a constant step size. *SIAM Journal on Optimization*, 18(1):29–51, 2007.
- [18] A. Bordes, L. Bottou, and P. Gallinari. SGD-QN: Careful Quasi-Newton Stochastic Gradient Descent. *Journal of Machine Learning Research*, 10:1737–1754, 2009.
- [19] Antoine Bordes, Léon Bottou, Patrick Gallinari, Jonathan Chang, and S. Alex Smith. Erratum: Sgdqn is less careful than expected. *Journal of Machine Learning Research*, 11:2229–2240, 2010.
- [20] L. Bottou. Stochastic Gradient Learning in Neural Networks. In *Proceedings of Neuro-Nîmes 91*, Nîmes, France, 1991. EC2.
- [21] L. Bottou. Online Algorithms and Stochastic Approximations. In D. Saad, editor, *Online Learning and Neural Networks*. Cambridge University Press, Cambridge, UK, 1998.
- [22] L. Bottou. Large-Scale Machine Learning with Stochastic Gradient Descent. In Y. Lechevallier and G. Saporta, editors, *Proceedings of the 19th International Conference on Computational Statistics (COMPSTAT'2010)*, pages 177–187, Paris, France, August 2010. Springer.
- [23] L. Bottou and O. Bousquet. The Tradeoffs of Large Scale Learning. In J. C. Platt, D. Koller, Y. Singer, and S. T. Roweis, editors, *Advances in Neural Information Processing Systems 20*, pages 161–168. Curran Associates, Inc., 2008.
- [24] L. Bottou, F. Fogelman Soulié, P. Blanchet, and J. S. Lienard. Experiments with Time Delay Networks and Dynamic Time Warping for Speaker Independent Isolated Digit Recognition. In *Proceedings of EuroSpeech 89*, volume 2, pages 537–540, Paris, France, 1989.
- [25] L. Bottou and Y. Le Cun. On-line Learning for Very Large Datasets. *Applied Stochastic Models in Business and Industry*, 21(2):137–151, 2005.
- [26] O. Bousquet. *Concentration Inequalities and Empirical Processes Theory Applied to the Analysis of Learning Algorithms*. PhD thesis, Ecole Polytechnique, 2002.
- [27] C. G. Broyden. The Convergence of a Class of Double-Rank Minimization Algorithms. *Journal of the Institute of Mathematics and Its Applications*, 6(1):76–90, 1970.
- [28] R. H. Byrd, G. M. Chin, J. Nocedal, and Y. Wu. Sample Size Selection in Optimization Methods for Machine Learning. *Mathematical Programming, Series B*, 134(1):127–155, 2012.
- [29] Richard H Byrd, Gillian M Chin, Will Neveitt, and Jorge Nocedal. On the use of stochastic Hessian information in optimization methods for machine learning. *SIAM Journal on Optimization*, 21(3):977–995, 2011.
- [30] Richard H Byrd, Gillian M Chin, Jorge Nocedal, and Figen Oztoprak. A family of second-order methods for convex  $\ell_1$ -regularized optimization. *Mathematical Programming*, pages 1–33, 2012.
- [31] Richard H Byrd, Jorge Nocedal, and Figen Oztoprak. An inexact successive quadratic approximation method for convex  $L_1$  regularized optimization. *arXiv preprint arXiv:1309.3529*, 2013.


---

## Page 39

- [32] Emmanuel J Candès and Benjamin Recht. Exact matrix completion via convex optimization. *Foundations of Computational mathematics*, 9(6):717–772, 2009.
- [33] Jean-François Cardoso. Blind signal separation: statistical principles. *Proc. of the IEEE*, 9(10):2009–2025, Oct 1998.
- [34] N. Cesa-Bianchi, A. Conconi, and C. Gentile. On the Generalization Ability of On-Line Learning Algorithms. *IEEE Transactions on Information Theory*, 50(9):2050–2057, 2004.
- [35] Anna Choromanska, Mikael Henaff, Michael Mathieu, Gérard Ben Arous, and Yann LeCun. The loss surfaces of multilayer networks. In *AISTATS*, 2015.
- [36] K. L. Chung. On a stochastic approximation method. *Annals of Mathematical Statistics*, 25(3):463–483, 1954.
- [37] A. R. Conn, N. I. M. Gould, and Ph. L. Toint. *Trust Region Methods*. Society for Industrial and Applied Mathematics, 2000.
- [38] C. Cortes and V. N. Vapnik. Support-Vector Networks. *Machine Learning*, 20(3):273–297, 1995.
- [39] R. Courant and H. Robbins. *What is Mathematics?* Oxford University Press, First edition, 1941.
- [40] R. Courant and H. Robbins. *What is Mathematics?* Oxford University Press, Second edition, 1996. Revised by I. Stewart.
- [41] T. M. Cover. Universal Portfolios. *Mathematical Finance*, 1(1):1–29, 1991.
- [42] I. Daubechies, M. Defrise, and C. De Mol. An iterative thresholding algorithm for linear inverse problems with a sparsity constraint. *Communications on Pure and Applied Mathematics*, 58:1413–1457, 2004.
- [43] Yann N. Dauphin, Harm de Vries, Junyoung Chung, and Yoshua Bengio. Rmsprop and equilibrated adaptive learning rates for non-convex optimization. *CoRR*, abs/1502.04390, 2015.
- [44] Yann N. Dauphin, Razvan Pascanu, Çaglar Gülçehre, KyungHyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. In *Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada*, pages 2933–2941, 2014.
- [45] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, Q. V. Le, M. Z. Mao, M. Ranzato, A. W. Senior, P. A. Tucker, K. Yang, and A. Y. Ng. Large Scale Distributed Deep Networks. In *Advances in Neural Information Processing Systems 25*, pages 1232–1240, 2012.
- [46] A. Defazio, F. Bach, and S. Lacoste-Julien. SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, *Advances in Neural Information Processing Systems 27*, pages 1646–1654. Curran Associates, Inc., 2014.


---

## Page 40

- [47] R. S. Dembo, Eisenstat S. C., and T. Steihaug. Inexact Newton Methods. *SIAM Journal on Numerical Analysis*, 19(2):400–408, 1982.
- [48] A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum Likelihood from Incomplete Data via the EM Algorithm. *Journal of the Royal Statistical Society. Series B (Methodological)*, 39(1):1–38, 1977.
- [49] Jia Deng, Nan Ding, Yangqing Jia, Andrea Frome, Kevin Murphy, Samy Bengio, Yuan Li, Hartmut Neven, and Hartwig Adam. Large-scale object classification using label relation graphs. In *Proceedings of the 13th European Conference on Computer Vision (ECCV), Part I*, pages 48–64, 2014.
- [50] L. Deng, G. E. Hinton, and B. Kingsbury. New types of deep neural network learning for speech recognition and related applications: An overview. In *IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2013, Vancouver, BC, Canada*, pages 8599–8603, 2013.
- [51] J. E. Dennis and J. J. Moré. A Characterization of Superlinear Convergence and Its Application to Quasi-Newton Methods. *Mathematics of Computation*, 28(126):549–560, 1974.
- [52] J. E. Dennis, Jr. and R. B. Schnabel. *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*. Classics in Applied Mathematics. SIAM, Philadelphia, PA, USA, 1996.
- [53] D.L. Donoho. De-noising by soft-thresholding. *Information Theory, IEEE Transactions on*, 41(3):613–627, 1995.
- [54] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research*, 12:2121–2159, 2011.
- [55] R. M. Dudley. *Uniform Central Limit Theorems*. Cambridge Studies in Advanced Mathematics, 63. Cambridge University Press, 1999.
- [56] S. T. Dumais, J. C. Platt, D. Hecherman, and M. Sahami. Inductive Learning Algorithms and Representations for Text Categorization. In *Proceedings of the 1998 ACM CIKM International Conference on Information and Knowledge Management, Bethesda, Maryland, USA*, pages 148–155, 1998.
- [57] J. Eckstein and D. P. Bertsekas. On the Douglas-Rachford splitting method and the proximal point algorithm for maximal monotone operators. *Mathematical Programming*, 55:293–318, 1992.
- [58] Rong-En Fan, Kai-Wei Chang, Cho-Jui Hsieh, Xiang-Rui Wang, and Chih-Jen Lin. Liblinear: A library for large linear classification. *The Journal of Machine Learning Research*, 9:1871–1874, 2008.
- [59] M. A. T. Figueiredo, R. D. Nowak, and S. J. Wright. Gradient Projection for Sparse Reconstruction: Application to Compressed Sensing and Other Inverse Problems. *IEEE Journal of Selected Topics in Signal Processing*, 1(4):586–597, 2007.


---

## Page 41

- [60] R. Fletcher. A New Approach to Variable Metric Algorithms. *Computer Journal*, 13(3):317–322, 1970.
- [61] J. E. Freund. *Mathematical Statistics*. Prentice-Hall, Englewood Cliffs, NJ, USA, 1962.
- [62] M. P. Friedlander and M. Schmidt. Hybrid Deterministic-Stochastic Methods for Data Fitting. *SIAM Journal on Scientific Computing*, 34(3):A1380–A1405, 2012.
- [63] M. C. Fu. Optimization for simulation: Theory vs. practice. *INFORMS Journal on Computing*, 14(3):192–215, 2002.
- [64] D. Gabay and B. Mercier. A Dual Algorithm for the Solution of Nonlinear Variational Problems via Finite Element Approximations. *Computers and Mathematics with Applications*, 2:17–40, 1976.
- [65] Gilles Gasso, Aristidis Pappaioannou, Marina Spivak, and Léon Bottou. Batch and online learning algorithms for nonconvex Neyman-Pearson classification. *ACM Transaction on Intelligent System and Technologies*, 3(2), 2011.
- [66] E. G. Gladyshev. On stochastic approximations. *Theory of Probability and its Applications*, 10:275–278, 1965.
- [67] R. Glowinski and A. Marrocco. Sur l’approximation, par éléments finis d’ordre un, et la résolution, par pénalisation-dualité, d’une classe de problèmes de Dirichlet non linéaires. *Revue Française d’Automatique, Informatique, et Recherche Opérationnelle*, 9:41–76, 1975.
- [68] D. Goldfarb. A Family of Variable Metric Updates Derived by Variational Means. *Mathematics of Computation*, 24(109):23–26, 1970.
- [69] G. H. Golub and C. F. Van Loan. *Matrix Computations*. Johns Hopkins University Press, Baltimore, MD, USA, fourth edition, 2012.
- [70] Ian J. Goodfellow and Oriol Vinyals. Qualitatively characterizing neural network optimization problems. *CoRR*, abs/1412.6544, 2014.
- [71] A. Griewank. *Automatic Differentiation*. Princeton Companion to Applied Mathematics, Nicolas Higham Ed., Princeton University Press, 2014.
- [72] Mert Gurbuzbalaban, Asuman Ozdaglar, and Pablo Parrilo. On the convergence rate of incremental aggregated gradient algorithms. *ArXiv*, abs/1506.02081, 2015.
- [73] F. S. Hashemi, S. Ghosh, and R. Pasupathy. On adaptive sampling rules for stochastic recursions. In *Simulation Conference (WSC), 2014 Winter*, pages 3959–3970, 2014.
- [74] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
- [75] W. Hoeffding. Probability Inequalities for Sums of Bounded Random Variables. *Journal of the American Statistical Association*, 58(301):13–30, 1963.


---

## Page 42

- [76] Matthew D. Hoffman, David M. Blei, Chong Wang, and John William Paisley. Stochastic variational inference. *Journal of Machine Learning Research*, 14(1):1303–1347, 2013.
- [77] T. Homem-de Mello and A. Shapiro. A Simulation-Based Approach to Two-Stage Stochastic Programming with Recourse. *Mathematical Programming*, 81(3):301–325, 1998.
- [78] C. J. Hsieh, M. A. Sustik, P. Ravikumar, and I. S. Dhillon. Sparse inverse covariance matrix estimation using quadratic approximation. *Advances in Neural Information Processing Systems (NIPS)*, 24, 2011.
- [79] Cho-Jui Hsieh, Mátyás A Sustik, Inderjit S Dhillon, Pradeep K Ravikumar, and Russell Poldrack. Big & quic: Sparse inverse covariance estimation for a million variables. In *Advances in Neural Information Processing Systems*, pages 3165–3173, 2013.
- [80] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015*, pages 448–456, 2015.
- [81] T. Joachims. Text Categorization with Support Vector Machines: Learning with Many Relevant Features. In *Proceedings of the 10th European Conference on Machine Learning (ECML'98), Chemnitz, Germany*, pages 137–142, 1998.
- [82] Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In *Advances in Neural Information Processing Systems*, pages 315–323, 2013.
- [83] R. E. Kalman. A New Approach to Linear Filtering and Prediction Problems. *Transactions of the ASME – Journal of Basic Engineering (Series D)*, 82:35–45, 1960.
- [84] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *CoRR*, abs/1412.6980, 2014.
- [85] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In *Advances in Neural Information Processing Systems 25*, 2012.
- [86] Jean Lafond, Nicolas Vasilache, and Léon Bottou. Manuscript in preparation, 2016.
- [87] Y. Le Cun, B. E. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. E. Hubbard, and L. D. Jackel. Handwritten Digit Recognition with a Back-Propagation Network. In *Advances in Neural Information Processing Systems 2*, pages 396–404, 1989.
- [88] Y. Le Cun, L. Bottou, Y. Bengio, and P. Haffner. Gradient Based Learning Applied to Document Recognition. *Proceedings of IEEE*, 86(11):2278–2324, 1998.
- [89] Y. Le Cun, L. Bottou, G. B. Orr, and K.-R. Müller. Efficient Backprop. In *Neural Networks, Tricks of the Trade*, Lecture Notes in Computer Science LNCS 1524. Springer Verlag, 1998.
- [90] N. Le Roux, M. Schmidt, and F. R. Bach. A Stochastic Gradient Method with an Exponential Convergence Rate for Finite Training Sets. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, *Advances in Neural Information Processing Systems 25*, pages 2663–2671. Curran Associates, Inc., 2012.


---

## Page 43

- [91] W. S. Lee, P. L. Bartlett, and R. C. Williamson. The Importance of Convexity in Learning with Squared Loss. *IEEE Transactions on Information Theory*, 44(5):1974–1980, 1998.
- [92] Todd K. Leen and Genevieve B. Orr. Optimal stochastic search and adaptive momentum. In *Advances in Neural Information Processing Systems 6, [7th NIPS Conference, Denver, Colorado, USA, 1993]*, pages 477–484, 1993.
- [93] Dennis Leventhal and Adrian S Lewis. Randomized methods for linear constraints: convergence rates and conditioning. *Mathematics of Operations Research*, 35(3):641–654, 2010.
- [94] D. D. Lewis, Y. Yang, T. G. Rose, and F. Li. RCV1: A New Benchmark Collection of Text Categorization Research. *Journal of Machine Learning Research*, 5:361–397, 2004.
- [95] Chih-Jen Lin and Jorge J Moré. Newton’s method for large bound-constrained optimization problems. *SIAM Journal on Optimization*, 9(4):1100–1127, 1999.
- [96] H. Lin, J. Mairal, and Z. Harchaoui. A universal catalyst for first-order optimization. *Advances in Neural Information Processing Systems (NIPS)*, 2015.
- [97] D. C. Liu and J. Nocedal. On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45:503–528, 1989.
- [98] J. Liu, S. J. Wright, C. Ré, V. Bittorf, and S. Sridhar. An Asynchronous Parallel Stochastic Coordinate Descent Algorithm. *Journal of Machine Learning Research*, 16:285–322, 2015.
- [99] Gaétan Marceau-Caron and Yann Ollivier. Practical riemannian neural networks. *CoRR*, abs/1602.08007, 2016.
- [100] J. Martens and Sutskever I. Learning Recurrent Neural Networks with Hessian-Free Optimization. In *28th International Conference on Machine Learning. ICML*, 2011.
- [101] James Martens. New insights and perspectives on the natural gradient method. *arXiv preprint arXiv:1412.1193*, 2014.
- [102] P. Massart. Some applications of concentration inequalities to Statistics. *Annales de la Faculté des Sciences de Toulouse*, series 6, 9(2):245–303, 2000.
- [103] A. Mokhtari and A. Ribeiro. RES: Regularized Stochastic BFGS algorithm. *IEEE Transactions on Signal Processing*, 62(23):6089–6104, 2014.
- [104] N. Murata. A Statistical Study of On-line Learning. In D. Saad, editor, *On-line Learning in Neural Networks*, pages 63–92. Cambridge University Press, New York, NY, USA, 1998.
- [105] A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro. Robust Stochastic Approximation Approach to Stochastic Programming. *SIAM Journal on Optimization*, 19(4):1574–1609, 2009.
- [106] A. S. Nemirovski and D. B. Yudin. On Cezaro’s convergence of the steepest descent method for approximating saddle point of convex-concave functions. *Soviet Mathematics–Doklady*, 19, 1978.


---

## Page 44

- [107] Y. Nesterov. A method of solving a convex programming problem with convergence rate  $\mathcal{O}(1/k^2)$ . *Soviet Mathematics Doklady*, 27(2):372–376, 1983.
- [108] Y. Nesterov. *Introductory Lectures on Convex Optimization: A Basic Course*. Kluwer Academic Publishers, 2004.
- [109] Y. Nesterov. Primal-dual subgradient methods for convex problems. *Mathematical Programming, Series B*, 120(1):221–259, 2009.
- [110] Yu Nesterov. Efficiency of coordinate descent methods on huge-scale optimization problems. *SIAM Journal on Optimization*, 22(2):341–362, 2012.
- [111] NIPS Foundation. Advances in neural information processing systems (29 volumes), 1987–2016. Volumes 0–28, <http://papers.nips.cc>.
- [112] F. Niu, B. Recht, C. Re, and S. J. Wright. Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent. In *Advances in Neural Information Processing Systems 24*, pages 693–701, 2011.
- [113] J. Nocedal. Updating Quasi-Newton Matrices With Limited Storage. *Mathematics of Computation*, 35(151):773–782, 1980.
- [114] J. Nocedal and S. J. Wright. *Numerical Optimization*. Springer New York, Second edition, 2006.
- [115] A. B. J. Novikoff. On Convergence Proofs on Perceptrons. In *Proceedings of the Symposium on the Mathematical Theory of Automata*, volume 12, pages 615–622, 1962.
- [116] J. Nutini, M. Schmidt, I. H. Laradji, M. Friedlander, and H. Koepke. Coordinate Descent Converges Faster with the Gauss-Southwell Rule than Random Selection. In *32nd International Conference on Machine Learning*. ICML, 2015.
- [117] J. Ortega and W. Rheinboldt. *Iterative Solution of Nonlinear Equations in Several Variables*. Society for Industrial and Applied Mathematics, 2000.
- [118] M. R. Osborne. Fisher’s method of scoring. *International Statistical Review / Revue Internationale de Statistique*, 60(1):99–117, Apr 1992.
- [119] Hyeoung Park, Sun-ichi Amari, and Kenji Fukumizu. Adaptive natural gradient learning algorithms for various stochastic models. *Neural Networks*, 13(7):755–764, 2000.
- [120] R. Pasupathy, P. W. Glynn, S. Ghosh, and F. S. Hashemi. On Sampling Rates in Stochastic Recursions. 2015. Under review.
- [121] B. A. Pearlmutter. Fast Exact Multiplication by the Hessian. *Neural Computation*, 6(1):147–160, 1994.
- [122] Mert Pilanci and Martin J Wainwright. Newton sketch: A linear-time optimization algorithm with linear-quadratic convergence. *arXiv preprint arXiv:1505.02250*, 2015.
- [123] B. T. Polyak. Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5):1–17, 1964.


---

## Page 45

- [124] B. T. Polyak. Comparison of the convergence rates for single-step and multi-step optimization algorithms in the presence of noise. *Engineering Cybernetics*, 15(1):6–10, 1977.
- [125] B. T. Polyak. New Method of Stochastic Approximation Type. *Automation and Remote Control*, 51(4):937–946, 1991.
- [126] B. T. Polyak and A. B. Juditsky. Acceleration of Stochastic Approximation by Averaging. *SIAM Journal on Control and Optimization*, 30(4):838–855, 1992.
- [127] M. J. D. Powell. On Search Directions for Minimization Algorithms. *Mathematical Programming*, 4(1):193–201, 1973.
- [128] Maxim Raginsky and Alexander Rakhlin. Information-based complexity, feedback and dynamics in convex programming. *IEEE Transactions on Information Theory*, 57(10):7036–7056, 2011.
- [129] A. Razavian, H. Azizpour, J. Sullivan, and S. Carlsson. CNN features off-the-shelf: an astounding baseline for recognition. *arXiv:1403.6382*, 2014.
- [130] H. Robbins and S. Monro. A Stochastic Approximation Method. *The Annals of Mathematical Statistics*, 22(3):400–407, 1951.
- [131] H. Robbins and D. Siegmund. A convergence theorem for non negative almost supermartingales and some applications. In Jagdish S. Rustagi, editor, *Optimizing Methods in Statistics*. Academic Press, 1971.
- [132] Farbod Roosta-Khorasani and Michael W Mahoney. Sub-sampled Newton methods II: Local convergence rates. *arXiv preprint arXiv:1601.04738*, 2016.
- [133] Stéphane Ross, Paul Mineiro, and John Langford. Normalized online learning. In *Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence (UAI)*, 2013.
- [134] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning internal representations by error propagation. In *Parallel Distributed Processing: Explorations in the Microstructure of Cognition*, volume 1, chapter 8, pages 318–362. MIT Press, Cambridge, MA, 1986.
- [135] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning Representations by Back-Propagating Errors. *Nature*, 323:533–536, 1986.
- [136] D. Ruppert. Efficient estimations from a slowly convergent Robbins-Monro process. Technical report, Cornell University Operations Research and Industrial Engineering, 1988.
- [137] Katya Scheinberg and Xiaocheng Tang. Practical inexact proximal quasi-newton method with global complexity analysis. *arXiv preprint arXiv:1311.6547*, 2013.
- [138] M. Schmidt. *Graphical Model Structure Learning with  $\ell_1$ -Regularization*. PhD thesis, University of British Columbia, 2010.
- [139] M. Schmidt, N. Le Roux, and F. Bach. Minimizing finite sums with the stochastic average gradient. *arXiv preprint arXiv:1309.2388*, 2013.


---

## Page 46

- [140] Mark Schmidt, Nicolas L Roux, and Francis R Bach. Convergence rates of inexact proximal-gradient methods for convex optimization. In *Advances in neural information processing systems*, pages 1458–1466, 2011.
- [141] N. N. Schraudolph. Fast Curvature Matrix-Vector Products. In G. Dorffner, H. Bischof, and K. Hornik, editors, *Artificial Neural Networks, ICANN 2001*, volume 2130 of *Lecture Notes in Computer Science*, pages 19–26. Springer-Verlag Berlin Heidelberg, 2001.
- [142] N. N. Schraudolph, J. Yu, and S. Günter. A stochastic quasi-Newton method for online convex optimization. In *International Conference on Artificial Intelligence and Statistics*, pages 436–443. Society for Artificial Intelligence and Statistics, 2007.
- [143] G. Shafer and V. Vovk. *Probability and finance: It’s only a game!*, volume 491. John Wiley & Sons, 2005.
- [144] S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter. Pegasos: primal estimated sub-gradient solver for SVM. *Mathematical Programming*, 127(1):3–30, 2011.
- [145] S. Shalev-Shwartz and T. Zhang. Stochastic dual coordinate ascent methods for regularized loss. *Journal of Machine Learning Research*, 14(1):567–599, 2013.
- [146] D. F. Shanno. Conditioning of Quasi-Newton Methods for Function Minimization. *Mathematics of Computation*, 24(111):647–656, 1970.
- [147] S. Sra, S. Nowozin, and S.J. Wright. *Optimization for Machine Learning*. Mit Pr, 2011.
- [148] Stanford Vision Lab. ImageNet Large Scale Visual Recognition Challenge (ILSVRC). <http://image-net.org/challenges/LSVRC/>, 2015 (accessed October 25, 2015).
- [149] T. Steihaug. The Conjugate Gradient Method and Trust Regions in Large Scale Optimization. *SIAM Journal on Numerical Analysis*, 20(3):626–637, 1983.
- [150] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in deep learning. In *30th International Conference on Machine Learning*. ICML, 2013.
- [151] Robert Tibshirani. Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society. Series B (Methodological)*, pages 267–288, 1996.
- [152] Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5. RMSPROP: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012.
- [153] P. Tseng and S. Yun. A coordinate gradient descent method for nonsmooth separable minimization. *Mathematical Programming*, 117(1-2):387–423, 2009.
- [154] A. B. Tsybakov. Optimal aggregation of classifiers in statistical learning. *Annals of Statistics*, 32(1), 2004.
- [155] V. N. Vapnik. *Estimation of Dependences Based on Empirical Data*. Springer, 1983.


---

## Page 47

- [156] V. N. Vapnik. *Statistical Learning Theory*. Wiley, New York, 1998.
- [157] V. N. Vapnik and A. Y. Chervonenkis. *Theory of Pattern Recognition*. Nauka, Moscow, 1974.  
  German Translation: *Theorie der Zeichenerkennung*, Akademie-Verlag, Berlin, 1979.
- [158] V. N. Vapnik and A. Ya. Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. *Proceedings of the USSR Academy of Sciences*, 181(4):781–783, 1968. English translation: *Soviet Math. Dokl.*, 9:915–918, 1968.
- [159] S. J. Wright, R. D. Nowak, and M. A. T. Figueiredo. Sparse Reconstruction by Separable Approximation. *IEEE Transactions on Signal Processing*, 57(7):2479–2493, 2009.
- [160] C. F. J. Wu. On the convergence properties of the em algorithm. *The Annals of Statistics*, 11(1):95–103, 1983.
- [161] W. Xu. Towards optimal one pass large scale learning with averaged stochastic gradient descent. *arXiv preprint arXiv:1107.2490v2*, 2011.
- [162] Matthew D. Zeiler. ADADELTA: an adaptive learning rate method. *CoRR*, abs/1212.5701, 2012.
- [163] C. Zhu, R. H. Byrd, P. Lu, and J. Nocedal. Algorithm 78: L-BFGS-B: Fortran subroutines for large-scale bound constrained optimization. *ACM Transactions on Mathematical Software*, 23(4):550–560, 1997.
- [164] M. Zinkevich. Online Convex Programming and Generalized Infinitesimal Gradient Ascent. In *Proceedings of the Twentieth International Conference on Machine Learning (ICML)*, pages 928–936, 2003.
