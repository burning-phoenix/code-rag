/-
# TropicalNN.Perceptron — Single ReLU Neuron as Tropical Rational Function

The key identity: max(a - b, 0) = max(a, b) - b
This converts ReLU(F(x) - G(x)) into a tropical rational form.

## Reference
tropical_perceptron_derivation.md
-/

import TropicalNN.Defs

set_option autoImplicit false

namespace TropicalNN

/-! ## The fundamental ReLU-tropical identity -/

/-- **The fundamental identity connecting ReLU to tropical algebra.**
    max(a - b, 0) = max(a, b) - b.
    This is what makes ReLU networks tropical rational maps. -/
theorem relu_tropical_identity (a b : ℝ) :
    max (a - b) 0 = max a b - b := by
  rw [show (0 : ℝ) = b - b from (sub_self b).symm, max_sub_sub_right]

/-- Equivalent form: ReLU(a - b) = (a ⊕ₜ b) ⊘ₜ b -/
theorem relu_eq_tropAdd_tropDiv (a b : ℝ) :
    max (a - b) 0 = tropDiv (tropAdd a b) b := by
  simp [tropDiv, tropAdd, relu_tropical_identity]

/-! ## Weight decomposition -/

/-- Any real number can be decomposed as w = w⁺ - w⁻ where w⁺, w⁻ ≥ 0 -/
theorem weight_decomp (w : ℝ) :
    w = max w 0 - max (-w) 0 := by
  exact (max_zero_sub_eq_self w).symm

/-- The positive part is non-negative -/
theorem pos_part_nonneg (w : ℝ) : 0 ≤ max w 0 := by
  exact le_max_right w 0

/-- The negative part is non-negative -/
theorem neg_part_nonneg (w : ℝ) : 0 ≤ max (-w) 0 := by
  exact le_max_right (-w) 0

/-! ## Perceptron as tropical rational function

For a single ReLU neuron with weights w, bias b, input x:
  z = ∑ᵢ wᵢxᵢ + b = F(x) - G(x)
where F(x) = ∑ᵢ wᵢ⁺xᵢ + b, G(x) = ∑ᵢ wᵢ⁻xᵢ
(tropical monomials in standard arithmetic).

Then: y = ReLU(z) = max(F(x), G(x)) - G(x)
which is a tropical rational function.
-/

/-- The pre-activation of a perceptron decomposes into F - G
    where F uses positive parts and G uses negative parts of weights. -/
theorem preactivation_decomp {d : ℕ}
    (w : Fin d → ℝ) (b : ℝ) (x : Fin d → ℝ)
    (wpos : Fin d → ℝ) (wneg : Fin d → ℝ)
    (hw : ∀ i, w i = wpos i - wneg i)
    (hpos : ∀ i, 0 ≤ wpos i)
    (hneg : ∀ i, 0 ≤ wneg i) :
    (∑ i, w i * x i) + b =
    ((∑ i, wpos i * x i) + b) - (∑ i, wneg i * x i) := by
  have : ∑ i, w i * x i = ∑ i, (wpos i - wneg i) * x i := by
    congr 1; ext i; rw [hw i]
  rw [this]
  simp_rw [sub_mul]
  rw [Finset.sum_sub_distrib]
  ring

/-- The ReLU output of a perceptron has the tropical rational form:
    y = max(F(x), G(x)) - G(x) -/
theorem perceptron_tropical_form {d : ℕ}
    (F G : (Fin d → ℝ) → ℝ) (x : Fin d → ℝ) :
    max (F x - G x) 0 = max (F x) (G x) - G x := by
  exact relu_tropical_identity (F x) (G x)

/-! ## Decision boundary -/

/-- The decision boundary of a single ReLU neuron (where z = 0)
    is precisely where F(x) = G(x), i.e., the tropical hypersurface. -/
theorem relu_boundary_iff {d : ℕ}
    (F G : (Fin d → ℝ) → ℝ) (x : Fin d → ℝ) :
    F x - G x = 0 ↔ F x = G x := sub_eq_zero

end TropicalNN
