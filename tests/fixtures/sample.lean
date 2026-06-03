import Mathlib.Data.Real.Basic

/- A block comment that mentions def, theorem, and lemma keywords which must
   NOT be picked up as declarations because they are inside a comment.
   /- nested block -/ still inside the outer comment. -/

def double (n : Nat) : Nat := 2 * n

-- a line comment mentioning theorem should also be ignored
theorem double_eq (n : Nat) : double n = n + n := by
  simp [double, Nat.two_mul]

lemma double_zero : double 0 = 0 := by
  rfl
