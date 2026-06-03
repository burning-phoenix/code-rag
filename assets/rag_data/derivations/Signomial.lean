/-
# TropicalNN.Signomial — Tropical Signomial Evaluation and Properties

This file defines evaluation of tropical signomials (max of affine functions)
and proves foundational properties needed for convexity and piecewise-linearity.

## Key results:
- `TropSignomial.eval`: signomial evaluation as max over monomial evaluations
- `monomial_eval_affine`: monomial evaluation satisfies the affine equality

## Reference
tropical_perceptron_derivation.md, tropical_mlp_derivation.md
-/

import TropicalNN.Defs

set_option autoImplicit false

namespace TropicalNN

/-! ## Tropical signomial evaluation

We define evaluation via `List.foldl max` starting from the head.
-/

/-- Evaluate a tropical signomial at a point: max over all monomial evaluations.
    f(x) = max_{k ∈ [m]} (cₖ + αₖ · x) -/
noncomputable def TropSignomial.eval {d : ℕ} (s : TropSignomial d) (x : Fin d → ℝ) : ℝ :=
  match s.monomials, s.nonempty with
  | m :: rest, _ => rest.foldl (fun acc m' => max acc (m'.eval x)) (m.eval x)

/-- Evaluation of a single-monomial signomial equals that monomial's eval -/
theorem TropSignomial.eval_singleton {d : ℕ} (m : TropMonomial d) (hne : [m] ≠ [])
    (x : Fin d → ℝ) :
    (⟨[m], hne⟩ : TropSignomial d).eval x = m.eval x := by
  simp [TropSignomial.eval]

/-! ## Monomial evaluation is affine

The key property enabling convexity: each monomial c + α · x satisfies
the affine identity f(a • x + b • y) = a • f(x) + b • f(y) when a + b = 1.
-/

/-- Tropical monomial evaluation is affine: it preserves convex combinations. -/
theorem monomial_eval_affine {d : ℕ} (m : TropMonomial d)
    (a b : ℝ) (hab : a + b = 1)
    (x y : Fin d → ℝ) :
    m.eval (a • x + b • y) = a * m.eval x + b * m.eval y := by
  simp only [TropMonomial.eval, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  have key : ∀ i : Fin d, m.exponent i * (a * x i + b * y i) =
      a * (m.exponent i * x i) + b * (m.exponent i * y i) := fun i => by ring
  simp_rw [key, Finset.sum_add_distrib, ← Finset.mul_sum]
  set S₁ := ∑ i : Fin d, m.exponent i * x i
  set S₂ := ∑ i : Fin d, m.exponent i * y i
  have hc : m.coeff = a * m.coeff + b * m.coeff := by rw [← add_mul, hab, one_mul]
  conv_lhs => rw [hc]
  ring

/-- Monomial evaluation satisfies the convexity inequality (with equality) -/
theorem monomial_eval_convex_le {d : ℕ} (m : TropMonomial d)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1)
    (x y : Fin d → ℝ) :
    m.eval (a • x + b • y) ≤ a * m.eval x + b * m.eval y := by
  rw [monomial_eval_affine m a b hab x y]

/-! ## Tropical monomial is an affine function -/

/-- The evaluation of a tropical monomial c + α·x is an affine function of x -/
theorem tropMonomial_eval_eq {d : ℕ} (m : TropMonomial d) (x : Fin d → ℝ) :
    m.eval x = m.coeff + ∑ i, m.exponent i * x i := by
  rfl

/-! ## Foldl-max helper lemmas

These bridge the gap between the `List.foldl max` definition of signomial evaluation
and the mathematical properties needed for convexity and Jacobian proofs.
-/

/-- `List.foldl max init l ≥ init`: the fold never decreases below the initial value. -/
theorem foldl_max_ge_init (l : List ℝ) (init : ℝ) :
    l.foldl max init ≥ init := by
  induction l generalizing init with
  | nil => exact le_refl _
  | cons hd tl ih =>
    simp only [List.foldl_cons]
    exact le_trans (le_max_left init hd) (ih (max init hd))

/-- If `v ∈ l`, then `l.foldl max init ≥ v`. -/
theorem foldl_max_ge_of_mem (l : List ℝ) (init : ℝ) (v : ℝ) (hv : v ∈ l) :
    l.foldl max init ≥ v := by
  induction l generalizing init with
  | nil => contradiction
  | cons hd tl ih =>
    simp only [List.foldl_cons]
    rcases List.mem_cons.mp hv with rfl | hmem
    · exact le_trans (le_max_right init v) (foldl_max_ge_init tl (max init v))
    · exact ih (max init hd) hmem

/-- If `init ≤ C` and all elements `≤ C`, then `l.foldl max init ≤ C`. -/
theorem foldl_max_le (l : List ℝ) (init C : ℝ) (hinit : init ≤ C)
    (hall : ∀ v ∈ l, v ≤ C) :
    l.foldl max init ≤ C := by
  induction l generalizing init with
  | nil => exact hinit
  | cons hd tl ih =>
    simp only [List.foldl_cons]
    apply ih
    · exact max_le hinit (hall hd (List.mem_cons.mpr (Or.inl rfl)))
    · intro v hv
      exact hall v (List.mem_cons.mpr (Or.inr hv))

/-- If `init ≥ v` for all `v ∈ l`, then `l.foldl max init = init`. -/
theorem foldl_max_eq_of_ge_all (l : List ℝ) (init : ℝ)
    (hall : ∀ v ∈ l, init ≥ v) :
    l.foldl max init = init := by
  induction l generalizing init with
  | nil => simp
  | cons hd tl ih =>
    simp only [List.foldl_cons]
    have ha : init ≥ hd := hall hd (List.mem_cons.mpr (Or.inl rfl))
    rw [max_eq_left ha]
    exact ih init (fun v hv => hall v (List.mem_cons.mpr (Or.inr hv)))

/-! ## Bridge: monomial foldl ↔ real foldl -/

/-- Foldl of `max acc (f a)` over a list equals foldl of `max` over the mapped list. -/
theorem foldl_max_map {α : Type*} (f : α → ℝ) (l : List α) (init : ℝ) :
    l.foldl (fun acc a => max acc (f a)) init = (l.map f).foldl max init := by
  induction l generalizing init with
  | nil => simp
  | cons a rest ih =>
    simp only [List.foldl_cons, List.map_cons]
    exact ih (max init (f a))

/-! ## Signomial evaluation bounds

These lift the foldl-max helpers to `TropSignomial.eval`, connecting `Fin` indices
with list membership via `List.get_mem`.
-/

/-- Signomial evaluation is at least as large as any individual monomial evaluation. -/
theorem TropSignomial.eval_ge_monomial {d : ℕ} (s : TropSignomial d)
    (x : Fin d → ℝ) (k : Fin s.monomials.length) :
    s.eval x ≥ (s.monomials.get k).eval x := by
  match s, k with
  | ⟨m :: rest, _⟩, ⟨0, _⟩ =>
    show rest.foldl (fun acc m' => max acc (m'.eval x)) (m.eval x) ≥ m.eval x
    rw [foldl_max_map]
    exact foldl_max_ge_init _ _
  | ⟨m :: rest, _⟩, ⟨n + 1, hn⟩ =>
    show rest.foldl (fun acc m' => max acc (m'.eval x)) (m.eval x) ≥
      (rest.get ⟨n, Nat.lt_of_succ_lt_succ hn⟩).eval x
    rw [foldl_max_map]
    apply foldl_max_ge_of_mem
    exact List.mem_map.mpr ⟨_, List.get_mem rest ⟨n, Nat.lt_of_succ_lt_succ hn⟩, rfl⟩

/-- If all monomial evaluations are ≤ C, then signomial evaluation is ≤ C. -/
theorem TropSignomial.eval_le_of_all_le {d : ℕ} (s : TropSignomial d)
    (x : Fin d → ℝ) (C : ℝ)
    (h : ∀ k : Fin s.monomials.length, (s.monomials.get k).eval x ≤ C) :
    s.eval x ≤ C := by
  match s with
  | ⟨m :: rest, _⟩ =>
    show rest.foldl (fun acc m' => max acc (m'.eval x)) (m.eval x) ≤ C
    rw [foldl_max_map]
    apply foldl_max_le
    · exact h ⟨0, Nat.zero_lt_succ _⟩
    · intro v hv
      obtain ⟨m', hm'mem, rfl⟩ := List.mem_map.mp hv
      obtain ⟨⟨i, hi⟩, rfl⟩ := List.mem_iff_get.mp hm'mem
      exact h ⟨i + 1, Nat.succ_lt_succ hi⟩

/-- If monomial k's evaluation dominates all others, signomial eval = k's eval. -/
theorem TropSignomial.eval_eq_of_dominant {d : ℕ} (s : TropSignomial d)
    (x : Fin d → ℝ) (k : Fin s.monomials.length)
    (hdom : ∀ j : Fin s.monomials.length,
      (s.monomials.get k).eval x ≥ (s.monomials.get j).eval x) :
    s.eval x = (s.monomials.get k).eval x := by
  apply le_antisymm
  · exact TropSignomial.eval_le_of_all_le s x _ hdom
  · exact TropSignomial.eval_ge_monomial s x k

end TropicalNN
