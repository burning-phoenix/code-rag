/-
# TropicalNN.MLP — Recursive Tropical Structure of MLP Layers

An MLP is a composition of tropical rational functions. Each layer transforms
a tropical rational pair (F, G) into a new pair (F', G') via:
  1. Linear transformation (weight decomposition into positive/negative parts)
  2. ReLU activation (applying the fundamental tropical identity)

## Key results:
- Max of two tropical rationals is tropical rational
- Pre-activation decomposes into H - G_new
- Layer activation preserves tropical rational form via relu_tropical_identity

## Reference
tropical_mlp_derivation.md
-/

import TropicalNN.Perceptron

set_option autoImplicit false

namespace TropicalNN

/-! ## Max of two tropical rationals -/

/-- **Max of two tropical rational expressions is tropical rational.**
    max(F₁ - G₁, F₂ - G₂) = max(F₁ + G₂, F₂ + G₁) - (G₁ + G₂).
    This identity underlies max pooling and composition of tropical rationals. -/
theorem max_tropical_rational_combine (F₁ G₁ F₂ G₂ : ℝ) :
    max (F₁ - G₁) (F₂ - G₂) = max (F₁ + G₂) (F₂ + G₁) - (G₁ + G₂) := by
  have h1 : F₁ - G₁ = (F₁ + G₂) - (G₁ + G₂) := by ring
  have h2 : F₂ - G₂ = (F₂ + G₁) - (G₁ + G₂) := by ring
  rw [h1, h2, max_sub_sub_right]

/-! ## MLP layer recurrence -/

section LayerRecurrence

variable {n_prev : ℕ}

/-- **Pre-activation decomposition for a single neuron.**
    Given weight row decomposed as Apos - Aneg and previous layer output F - G,
    the pre-activation ∑ⱼ (Apos_j - Aneg_j)(F_j - G_j) + b decomposes as H - G_new where:
      H     = ∑ Apos·F + ∑ Aneg·G + b
      G_new = ∑ Apos·G + ∑ Aneg·F -/
theorem preactivation_layer_decomp
    (Apos_row Aneg_row F G : Fin n_prev → ℝ) (b : ℝ) :
    (∑ j, (Apos_row j - Aneg_row j) * (F j - G j)) + b =
    ((∑ j, Apos_row j * F j) + (∑ j, Aneg_row j * G j) + b) -
    ((∑ j, Apos_row j * G j) + (∑ j, Aneg_row j * F j)) := by
  simp_rw [sub_mul, mul_sub, Finset.sum_sub_distrib]
  ring

/-- **Layer activation preserves tropical rational form.**
    ReLU(H - G) = max(H, G) - G, giving the new tropical rational pair
    (F_new, G) where F_new = max(H, G).
    Direct application of the fundamental ReLU-tropical identity. -/
theorem layer_activation_tropical_form (H G : ℝ) :
    max (H - G) 0 = max H G - G :=
  relu_tropical_identity H G

/-- **Full MLP layer recurrence: tropical rational form is preserved.**
    Given layer l with output F - G (component-wise), the next layer computes:
      ν_new = max(H, G_new) - G_new
    where H = ∑ Apos·F + ∑ Aneg·G + b and G_new = ∑ Apos·G + ∑ Aneg·F.
    This establishes the inductive step of ZNL Proposition 5.1. -/
theorem mlp_layer_tropical_rational
    (Apos_row Aneg_row F G : Fin n_prev → ℝ) (b : ℝ) :
    let H := (∑ j, Apos_row j * F j) + (∑ j, Aneg_row j * G j) + b
    let G_new := (∑ j, Apos_row j * G j) + (∑ j, Aneg_row j * F j)
    max ((∑ j, (Apos_row j - Aneg_row j) * (F j - G j)) + b) 0 =
    max H G_new - G_new := by
  intro H G_new
  rw [preactivation_layer_decomp]
  exact relu_tropical_identity H G_new

end LayerRecurrence

end TropicalNN
