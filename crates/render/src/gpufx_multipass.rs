//! Pass plans for GPU effect routes that need intermediate images.
//!
//! A plan is an ordered list of compute passes over numbered slots. Slot 0 is the source image and
//! is read-only; every other slot is an intermediate written by an earlier pass. The plan is pure
//! data: it is validated here, without a device, and the `filmcraft-gpu` crate allocates and runs it.
//! Every count and size comes from the project or the media, so all arithmetic is checked and every
//! violation is an explicit error. Nothing here panics.

/// Maximum number of passes in one plan. Bounds the work a hostile plan can request.
pub const MAX_PASSES: usize = 64;
/// Maximum number of slots (source included). Bounds the scratch memory a plan can request.
pub const MAX_SLOTS: usize = 8;
/// Inputs one pass may read. The kernel contract in `filmcraft-gpu` binds two.
pub const MAX_INPUTS: usize = 2;

/// Bytes per pixel of the intermediate format (Rgba32Float).
pub const SLOT_BYTES_PER_PIXEL: u64 = 16;

/// Index of a plan slot. Slot 0 is the source image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slot(pub u8);

impl Slot {
    /// The source image, which no pass may write.
    pub const SOURCE: Slot = Slot(0);
}

/// One compute pass: reads up to [`MAX_INPUTS`] slots and writes exactly one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pass {
    /// Kernel name, looked up by the GPU side. Must be non-empty.
    pub kernel: &'static str,
    /// Slots read by the pass, in binding order.
    pub inputs: Vec<Slot>,
    /// Slot written by the pass.
    pub output: Slot,
}

/// An ordered multi-pass route with its result slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultipassPlan {
    pub passes: Vec<Pass>,
    /// Slot holding the final image after the last pass.
    pub result: Slot,
}

/// Why a plan was refused. Each variant names the first violated rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultipassError {
    /// No passes, or the result slot is not produced by any pass.
    EmptyPlan,
    /// More than [`MAX_PASSES`] passes.
    TooManyPasses { passes: usize },
    /// A slot index is outside `0..MAX_SLOTS`.
    SlotOutOfRange { slot: u8 },
    /// A pass reads more than [`MAX_INPUTS`] slots.
    TooManyInputs { pass: usize },
    /// A pass has no kernel name.
    EmptyKernel { pass: usize },
    /// A pass writes the source slot.
    WritesSource { pass: usize },
    /// A pass reads the slot it writes. Compute passes cannot alias input and output.
    ReadsOwnOutput { pass: usize },
    /// A pass reads an intermediate that no earlier pass wrote.
    ReadBeforeWrite { pass: usize, slot: u8 },
    /// The result slot is not written by any pass.
    ResultNotWritten { slot: u8 },
    /// Byte count of the intermediates overflows `u64`.
    SizeOverflow,
}

impl MultipassPlan {
    /// Validates the plan's structure. Returns the number of distinct intermediate slots it writes.
    pub fn validate(&self) -> Result<usize, MultipassError> {
        if self.passes.is_empty() {
            return Err(MultipassError::EmptyPlan);
        }
        if self.passes.len() > MAX_PASSES {
            return Err(MultipassError::TooManyPasses { passes: self.passes.len() });
        }
        let in_range = |s: Slot| if usize::from(s.0) < MAX_SLOTS { Ok(()) } else { Err(MultipassError::SlotOutOfRange { slot: s.0 }) };
        let mut written = [false; MAX_SLOTS];
        written[usize::from(Slot::SOURCE.0)] = true;
        for (index, pass) in self.passes.iter().enumerate() {
            if pass.kernel.is_empty() {
                return Err(MultipassError::EmptyKernel { pass: index });
            }
            if pass.inputs.len() > MAX_INPUTS {
                return Err(MultipassError::TooManyInputs { pass: index });
            }
            in_range(pass.output)?;
            if pass.output == Slot::SOURCE {
                return Err(MultipassError::WritesSource { pass: index });
            }
            for &input in &pass.inputs {
                in_range(input)?;
                if input == pass.output {
                    return Err(MultipassError::ReadsOwnOutput { pass: index });
                }
                if !written[usize::from(input.0)] {
                    return Err(MultipassError::ReadBeforeWrite { pass: index, slot: input.0 });
                }
            }
            written[usize::from(pass.output.0)] = true;
        }
        in_range(self.result)?;
        if !written[usize::from(self.result.0)] || self.result == Slot::SOURCE {
            return Err(MultipassError::ResultNotWritten { slot: self.result.0 });
        }
        let mut distinct = [false; MAX_SLOTS];
        for pass in &self.passes {
            distinct[usize::from(pass.output.0)] = true;
        }
        Ok(distinct.iter().filter(|&&d| d).count())
    }

    /// Bytes of the intermediate slots this plan writes at `width` x `height`, or `None` when the
    /// validation fails or the arithmetic overflows. No cap: the device decides whether they fit.
    pub fn intermediate_bytes(&self, width: u32, height: u32) -> Result<u64, MultipassError> {
        let slots = self.validate()?;
        slot_bytes(width, height).and_then(|one| one.checked_mul(u64::try_from(slots).ok()?)).ok_or(MultipassError::SizeOverflow)
    }
}

/// Bytes of one Rgba32Float slot at `width` x `height`, or `None` on overflow.
pub fn slot_bytes(width: u32, height: u32) -> Option<u64> {
    u64::from(width).checked_mul(u64::from(height))?.checked_mul(SLOT_BYTES_PER_PIXEL)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(kernel: &'static str, inputs: &[u8], output: u8) -> Pass {
        Pass { kernel, inputs: inputs.iter().map(|&s| Slot(s)).collect(), output: Slot(output) }
    }

    fn plan(passes: Vec<Pass>, result: u8) -> MultipassPlan {
        MultipassPlan { passes, result: Slot(result) }
    }

    #[test]
    fn two_pass_chain_counts_distinct_intermediates() {
        let p = plan(vec![pass("blur_h", &[0], 1), pass("blur_v", &[1], 2), pass("mix", &[0, 2], 1)], 1);
        // slot 1 is written twice but is one slot
        assert_eq!(p.validate(), Ok(2));
        assert_eq!(p.intermediate_bytes(4, 3), Ok(2 * 4 * 3 * 16));
    }

    #[test]
    fn empty_plan_is_refused() {
        assert_eq!(plan(vec![], 1).validate(), Err(MultipassError::EmptyPlan));
    }

    #[test]
    fn too_many_passes_is_refused_before_any_walk() {
        let passes = (0..=MAX_PASSES).map(|_| pass("k", &[0], 1)).collect();
        assert_eq!(plan(passes, 1).validate(), Err(MultipassError::TooManyPasses { passes: MAX_PASSES + 1 }));
    }

    #[test]
    fn out_of_range_slots_are_refused() {
        assert_eq!(plan(vec![pass("k", &[0], 8)], 8).validate(), Err(MultipassError::SlotOutOfRange { slot: 8 }));
        assert_eq!(plan(vec![pass("k", &[9], 1)], 1).validate(), Err(MultipassError::SlotOutOfRange { slot: 9 }));
        assert_eq!(plan(vec![pass("k", &[0], 1)], 200).validate(), Err(MultipassError::SlotOutOfRange { slot: 200 }));
    }

    #[test]
    fn writing_the_source_is_refused() {
        assert_eq!(plan(vec![pass("k", &[0], 0)], 0).validate(), Err(MultipassError::WritesSource { pass: 0 }));
    }

    #[test]
    fn aliasing_input_and_output_is_refused() {
        let p = plan(vec![pass("k", &[0], 1), pass("k", &[1], 1)], 1);
        assert_eq!(p.validate(), Err(MultipassError::ReadsOwnOutput { pass: 1 }));
    }

    #[test]
    fn reading_an_unwritten_intermediate_is_refused() {
        let p = plan(vec![pass("k", &[2], 1)], 1);
        assert_eq!(p.validate(), Err(MultipassError::ReadBeforeWrite { pass: 0, slot: 2 }));
    }

    #[test]
    fn result_must_be_an_intermediate_some_pass_wrote() {
        assert_eq!(plan(vec![pass("k", &[0], 1)], 2).validate(), Err(MultipassError::ResultNotWritten { slot: 2 }));
        assert_eq!(plan(vec![pass("k", &[0], 1)], 0).validate(), Err(MultipassError::ResultNotWritten { slot: 0 }));
    }

    #[test]
    fn too_many_inputs_and_empty_kernel_are_refused() {
        assert_eq!(plan(vec![pass("k", &[0, 0, 0], 1)], 1).validate(), Err(MultipassError::TooManyInputs { pass: 0 }));
        assert_eq!(plan(vec![pass("", &[0], 1)], 1).validate(), Err(MultipassError::EmptyKernel { pass: 0 }));
    }

    #[test]
    fn byte_arithmetic_overflow_is_an_error_not_a_panic() {
        assert_eq!(slot_bytes(u32::MAX, u32::MAX), None);
        let p = plan(vec![pass("k", &[0], 1)], 1);
        assert_eq!(p.intermediate_bytes(u32::MAX, u32::MAX), Err(MultipassError::SizeOverflow));
    }

    #[test]
    fn no_resolution_cap_is_applied() {
        let p = plan(vec![pass("k", &[0], 1)], 1);
        // 8K UHD is far above any fixed cap; the device, not this crate, decides
        assert_eq!(p.intermediate_bytes(7680, 4320), Ok(7680 * 4320 * 16));
    }
}
