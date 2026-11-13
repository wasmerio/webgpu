//! Limit tiers: an adapter's limits rounded down to a few steps, for an
//! embedder whose guests should not learn which GPU they run on
//! ([`crate::WebGpuCtxBuilder::tiered_limits`]).
//!
//! An adapter's exact limits say a good deal about the GPU and its driver. A
//! guest that is shown a tier learns only which of a few classes the adapter
//! is in. Both native hosts tier here, on `webgpu.h`'s own structs, so a guest
//! sees the same steps whichever GPU library is underneath.
//!
//! The tiers, and the way they apply, are Dawn's (`ApplyLimitTiers` in
//! `src/dawn/native/Limits.cpp`, at the revision `scripts/build-dawn.sh`
//! pins). Limits that belong together form a group and move together: a group
//! takes the highest tier that every one of its limits reaches.
//!
//! A device's limits are the ones it was asked for, so nothing but the
//! adapter's are tiered. A request for more than the tier is refused
//! ([`exceeded`]), since granting it would say what the tier hides.

use crate::abi::structs::{Chained, CompatibilityModeLimits, Limits};

/// Which way a limit is better.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Better {
    /// A maximum: a larger value allows more.
    Higher,
    /// An alignment: a smaller value allows more.
    Lower,
}

impl Better {
    /// Whether `a` allows more than `b`.
    fn is(self, a: u64, b: u64) -> bool {
        match self {
            Better::Higher => a > b,
            Better::Lower => a < b,
        }
    }
}

struct Tiered {
    /// The member's name in `webgpu.h`.
    name: &'static str,
    /// The member's name in this crate's structs.
    #[cfg_attr(not(test), allow(dead_code))]
    field: &'static str,
    better: Better,
    /// What the member holds when a guest asks for nothing in particular
    /// (`WGPU_LIMIT_U32_UNDEFINED` or `WGPU_LIMIT_U64_UNDEFINED`).
    undefined: u64,
    get: fn(&Limits, &CompatibilityModeLimits) -> u64,
    set: fn(&mut Limits, &mut CompatibilityModeLimits, u64),
    /// From the compatibility-mode tier up. Every limit of a group has as
    /// many.
    tiers: &'static [u64],
}

macro_rules! tiered {
    (core, $field:ident : $ty:ty, $name:literal, $better:ident, [$($tier:expr),+ $(,)?]) => {
        Tiered {
            name: $name,
            field: stringify!($field),
            better: Better::$better,
            undefined: <$ty>::MAX as u64,
            get: |limits, _| limits.$field as u64,
            set: |limits, _, value| limits.$field = value as $ty,
            tiers: &[$($tier),+],
        }
    };
    (compat, $field:ident : $ty:ty, $name:literal, $better:ident, [$($tier:expr),+ $(,)?]) => {
        Tiered {
            name: $name,
            field: stringify!($field),
            better: Better::$better,
            undefined: <$ty>::MAX as u64,
            get: |_, compat| compat.$field as u64,
            set: |_, compat, value| compat.$field = value as $ty,
            tiers: &[$($tier),+],
        }
    };
}

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Dawn's limit groups. The first tier of each limit is compatibility mode's
/// default and the second the specification's, which every adapter of a core
/// device reaches.
#[rustfmt::skip]
static GROUPS: &[&[Tiered]] = &[
    &[
        tiered!(core, max_compute_workgroup_storage_size: u32, "maxComputeWorkgroupStorageSize", Higher, [16384, 16384, 32768, 49152, 65536]),
    ],
    &[
        tiered!(core, max_compute_invocations_per_workgroup: u32, "maxComputeInvocationsPerWorkgroup", Higher, [128, 256, 1024]),
        tiered!(core, max_compute_workgroup_size_x: u32, "maxComputeWorkgroupSizeX", Higher, [128, 256, 1024]),
        tiered!(core, max_compute_workgroup_size_y: u32, "maxComputeWorkgroupSizeY", Higher, [128, 256, 1024]),
        tiered!(core, max_compute_workgroup_size_z: u32, "maxComputeWorkgroupSizeZ", Higher, [64, 64, 64]),
        tiered!(core, max_compute_workgroups_per_dimension: u32, "maxComputeWorkgroupsPerDimension", Higher, [65535, 65535, 65535]),
    ],
    &[
        tiered!(core, max_storage_buffer_binding_size: u64, "maxStorageBufferBindingSize", Higher,
            [128 * MIB, 128 * MIB, 256 * MIB, 512 * MIB, GIB, 2 * GIB - 4, 4 * GIB - 4]),
    ],
    &[
        tiered!(core, max_buffer_size: u64, "maxBufferSize", Higher, [256 * MIB, 256 * MIB, GIB, 2 * GIB, 4 * GIB - 4]),
    ],
    &[
        tiered!(core, max_dynamic_uniform_buffers_per_pipeline_layout: u32, "maxDynamicUniformBuffersPerPipelineLayout", Higher, [8, 8, 10, 10]),
        tiered!(core, max_dynamic_storage_buffers_per_pipeline_layout: u32, "maxDynamicStorageBuffersPerPipelineLayout", Higher, [4, 4, 8, 8]),
        tiered!(core, max_sampled_textures_per_shader_stage: u32, "maxSampledTexturesPerShaderStage", Higher, [16, 16, 16, 48]),
        tiered!(core, max_samplers_per_shader_stage: u32, "maxSamplersPerShaderStage", Higher, [16, 16, 16, 16]),
        tiered!(core, max_storage_textures_per_shader_stage: u32, "maxStorageTexturesPerShaderStage", Higher, [4, 4, 8, 8]),
        tiered!(compat, max_storage_textures_in_fragment_stage: u32, "maxStorageTexturesInFragmentStage", Higher, [4, 4, 8, 8]),
        tiered!(compat, max_storage_textures_in_vertex_stage: u32, "maxStorageTexturesInVertexStage", Higher, [0, 4, 8, 8]),
        tiered!(core, max_uniform_buffers_per_shader_stage: u32, "maxUniformBuffersPerShaderStage", Higher, [12, 12, 12, 12]),
    ],
    &[
        tiered!(core, max_storage_buffers_per_shader_stage: u32, "maxStorageBuffersPerShaderStage", Higher, [8, 8, 10, 16]),
        tiered!(compat, max_storage_buffers_in_fragment_stage: u32, "maxStorageBuffersInFragmentStage", Higher, [4, 8, 10, 16]),
        tiered!(compat, max_storage_buffers_in_vertex_stage: u32, "maxStorageBuffersInVertexStage", Higher, [0, 8, 10, 16]),
    ],
    &[
        tiered!(core, max_color_attachment_bytes_per_sample: u32, "maxColorAttachmentBytesPerSample", Higher, [32, 32, 64, 128]),
    ],
    &[
        tiered!(core, max_inter_stage_shader_variables: u32, "maxInterStageShaderVariables", Higher, [15, 16, 28]),
    ],
    &[
        tiered!(core, max_texture_dimension_1d: u32, "maxTextureDimension1D", Higher, [4096, 8192, 16384]),
        tiered!(core, max_texture_dimension_2d: u32, "maxTextureDimension2D", Higher, [4096, 8192, 16384]),
        tiered!(core, max_texture_dimension_3d: u32, "maxTextureDimension3D", Higher, [2048, 2048, 2048]),
        tiered!(core, max_texture_array_layers: u32, "maxTextureArrayLayers", Higher, [256, 256, 2048]),
    ],
    &[
        tiered!(core, max_immediate_size: u32, "maxImmediateSize", Higher, [64, 64]),
    ],
    &[
        tiered!(core, max_bind_groups: u32, "maxBindGroups", Higher, [4, 4, 4]),
        tiered!(core, max_bind_groups_plus_vertex_buffers: u32, "maxBindGroupsPlusVertexBuffers", Higher, [24, 24, 24]),
        tiered!(core, max_bindings_per_bind_group: u32, "maxBindingsPerBindGroup", Higher, [1000, 1000, 1000]),
        tiered!(core, max_uniform_buffer_binding_size: u64, "maxUniformBufferBindingSize", Higher, [16384, 65536, 65536]),
        tiered!(core, min_uniform_buffer_offset_alignment: u32, "minUniformBufferOffsetAlignment", Lower, [256, 256, 256]),
        tiered!(core, min_storage_buffer_offset_alignment: u32, "minStorageBufferOffsetAlignment", Lower, [256, 256, 256]),
        tiered!(core, max_vertex_buffers: u32, "maxVertexBuffers", Higher, [8, 8, 8]),
        tiered!(core, max_vertex_attributes: u32, "maxVertexAttributes", Higher, [16, 16, 30]),
        tiered!(core, max_vertex_buffer_array_stride: u32, "maxVertexBufferArrayStride", Higher, [2048, 2048, 2048]),
        tiered!(core, max_color_attachments: u32, "maxColorAttachments", Higher, [4, 8, 8]),
    ],
];

/// Rounds an adapter's limits down to their tiers.
///
/// Each group is tried against its tiers from the highest down, and is held
/// to the first one all of its limits reach. Limits of a group that reaches
/// none of them (no adapter WebGPU allows has such a group) are left where
/// Dawn leaves them: the ones before the limit that falls short are held to
/// the lowest tier, the rest keep the adapter's values.
pub(crate) fn apply(limits: &mut Limits, compat: &mut CompatibilityModeLimits) {
    for group in GROUPS {
        'tier: for tier in (0..group[0].tiers.len()).rev() {
            for limit in *group {
                let step = limit.tiers[tier];
                let value = (limit.get)(limits, compat);
                if limit.better.is(step, value) {
                    continue 'tier;
                }
                if step != value {
                    (limit.set)(limits, compat, step);
                }
            }
            break;
        }
    }

    // The tiers of limits that depend on each other are not aligned, so a
    // tiered limit may come out above the one that bounds it. Dawn settles
    // that the same way (`NormalizeLimits`). It also holds some limits to
    // constants of its own there, which an adapter's limits already respect
    // on the Dawn host, and which the wgpu host does not have.
    compat.max_storage_buffers_in_vertex_stage = compat
        .max_storage_buffers_in_vertex_stage
        .min(limits.max_storage_buffers_per_shader_stage);
    compat.max_storage_textures_in_vertex_stage = compat
        .max_storage_textures_in_vertex_stage
        .min(limits.max_storage_textures_per_shader_stage);
    compat.max_storage_buffers_in_fragment_stage = compat
        .max_storage_buffers_in_fragment_stage
        .min(limits.max_storage_buffers_per_shader_stage);
    compat.max_storage_textures_in_fragment_stage = compat
        .max_storage_textures_in_fragment_stage
        .min(limits.max_storage_textures_per_shader_stage);
    limits.max_storage_buffer_binding_size = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
        & !3;
    limits.max_uniform_buffer_binding_size = limits
        .max_uniform_buffer_binding_size
        .min(limits.max_buffer_size);
}

/// Why a device with the limits `required` asks for may not be made on an
/// adapter whose tiered limits are `allowed`: the first limit that asks for
/// more than its tier. `None` when every limit is within what the guest was
/// told the adapter has.
///
/// The backend holds a request to the adapter's real limits, which a guest
/// could find by asking for more and more. This check keeps a request to the
/// limits it was shown.
pub(crate) fn exceeded(
    required: &Limits,
    allowed: &Limits,
    allowed_compat: &CompatibilityModeLimits,
) -> Option<String> {
    let nothing = CompatibilityModeLimits {
        max_storage_buffers_in_vertex_stage: u32::MAX,
        max_storage_textures_in_vertex_stage: u32::MAX,
        max_storage_buffers_in_fragment_stage: u32::MAX,
        max_storage_textures_in_fragment_stage: u32::MAX,
    };
    let required_compat = required
        .chain
        .items
        .iter()
        .find_map(|item| match item {
            Chained::CompatibilityModeLimits(compat) => Some(compat),
            _ => None,
        })
        .unwrap_or(&nothing);
    GROUPS.iter().copied().flatten().find_map(|limit| {
        let asked = (limit.get)(required, required_compat);
        let offered = (limit.get)(allowed, allowed_compat);
        (asked != limit.undefined && limit.better.is(asked, offered)).then(|| {
            format!(
                "Required limit {} is {asked}, and this adapter supports {offered}.",
                limit.name
            )
        })
    })
}

/// Limits as a guest leaves them when it asks for nothing in particular.
#[cfg(test)]
pub(crate) fn undefined() -> Limits {
    let mut limits = Limits::default();
    let mut compat = CompatibilityModeLimits::default();
    for limit in GROUPS.iter().copied().flatten() {
        (limit.set)(&mut limits, &mut compat, limit.undefined);
    }
    limits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every limit at its tier `index`, or at its highest if it has fewer.
    /// Tier 1 is the specification's default limits.
    fn tier(index: usize) -> (Limits, CompatibilityModeLimits) {
        let mut limits = Limits::default();
        let mut compat = CompatibilityModeLimits::default();
        for limit in GROUPS.iter().copied().flatten() {
            let step = limit.tiers[index.min(limit.tiers.len() - 1)];
            (limit.set)(&mut limits, &mut compat, step);
        }
        (limits, compat)
    }

    #[test]
    fn every_limit_of_webgpu_h_has_tiers() {
        // A member added to either struct fails to compile here until it is
        // given tiers above.
        let Limits {
            chain: _,
            max_texture_dimension_1d: _,
            max_texture_dimension_2d: _,
            max_texture_dimension_3d: _,
            max_texture_array_layers: _,
            max_bind_groups: _,
            max_bind_groups_plus_vertex_buffers: _,
            max_bindings_per_bind_group: _,
            max_dynamic_uniform_buffers_per_pipeline_layout: _,
            max_dynamic_storage_buffers_per_pipeline_layout: _,
            max_sampled_textures_per_shader_stage: _,
            max_samplers_per_shader_stage: _,
            max_storage_buffers_per_shader_stage: _,
            max_storage_textures_per_shader_stage: _,
            max_uniform_buffers_per_shader_stage: _,
            max_uniform_buffer_binding_size: _,
            max_storage_buffer_binding_size: _,
            min_uniform_buffer_offset_alignment: _,
            min_storage_buffer_offset_alignment: _,
            max_vertex_buffers: _,
            max_buffer_size: _,
            max_vertex_attributes: _,
            max_vertex_buffer_array_stride: _,
            max_inter_stage_shader_variables: _,
            max_color_attachments: _,
            max_color_attachment_bytes_per_sample: _,
            max_compute_workgroup_storage_size: _,
            max_compute_invocations_per_workgroup: _,
            max_compute_workgroup_size_x: _,
            max_compute_workgroup_size_y: _,
            max_compute_workgroup_size_z: _,
            max_compute_workgroups_per_dimension: _,
            max_immediate_size: _,
        } = Limits::default();
        let CompatibilityModeLimits {
            max_storage_buffers_in_vertex_stage: _,
            max_storage_textures_in_vertex_stage: _,
            max_storage_buffers_in_fragment_stage: _,
            max_storage_textures_in_fragment_stage: _,
        } = CompatibilityModeLimits::default();

        let names: Vec<_> = GROUPS
            .iter()
            .copied()
            .flatten()
            .map(|limit| limit.name)
            .collect();
        assert_eq!(names.len(), 32 + 4, "the members destructured above");
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "a limit is listed twice");
    }

    #[test]
    fn every_entry_reads_and_writes_the_member_it_is_named_after() {
        // `max_texture_dimension_2d` is `maxTextureDimension2D` in webgpu.h.
        let camel = |field: &str| {
            let mut name = String::new();
            for (index, word) in field.split('_').enumerate() {
                match (index, word) {
                    (0, _) => name.push_str(word),
                    (_, "1d" | "2d" | "3d") => name.push_str(&word.to_uppercase()),
                    _ => {
                        name.push_str(&word[..1].to_uppercase());
                        name.push_str(&word[1..]);
                    }
                }
            }
            name
        };
        let all: Vec<_> = GROUPS.iter().copied().flatten().collect();
        for (index, limit) in all.iter().enumerate() {
            assert_eq!(camel(limit.field), limit.name);
            // What one entry writes, only that entry reads.
            let mut limits = Limits::default();
            let mut compat = CompatibilityModeLimits::default();
            (limit.set)(&mut limits, &mut compat, 7);
            for (other, entry) in all.iter().enumerate() {
                assert_eq!(
                    (entry.get)(&limits, &compat),
                    if other == index { 7 } else { 0 },
                    "{} after writing {}",
                    entry.name,
                    limit.name
                );
            }
        }
    }

    #[test]
    fn a_group_that_reaches_no_tier_is_left_where_dawn_leaves_it() {
        // No adapter WebGPU allows is like this: 3D textures below even
        // compatibility mode's default.
        let (mut limits, mut compat) = tier(2);
        limits.max_texture_dimension_3d = 1024;
        apply(&mut limits, &mut compat);
        // The limits before the one that falls short: held to the lowest tier.
        assert_eq!(limits.max_texture_dimension_1d, 4096);
        assert_eq!(limits.max_texture_dimension_2d, 4096);
        // That one, and those after it: as the adapter has them.
        assert_eq!(limits.max_texture_dimension_3d, 1024);
        assert_eq!(limits.max_texture_array_layers, 2048);
        // Other groups are not affected.
        assert_eq!(limits.max_vertex_attributes, 30);
    }

    #[test]
    fn the_limits_of_a_group_have_as_many_tiers_and_never_get_worse() {
        for group in GROUPS {
            for limit in *group {
                assert_eq!(
                    limit.tiers.len(),
                    group[0].tiers.len(),
                    "{} and {}",
                    limit.name,
                    group[0].name
                );
                assert!(
                    limit
                        .tiers
                        .windows(2)
                        .all(|pair| !limit.better.is(pair[0], pair[1])),
                    "{}: a higher tier allows less",
                    limit.name
                );
            }
        }
    }

    #[test]
    fn limits_that_are_a_tier_stay_as_they_are() {
        for index in 1..7 {
            let (mut limits, mut compat) = tier(index);
            let (expected, expected_compat) = tier(index);
            apply(&mut limits, &mut compat);
            // Buffer sizes bound binding sizes, and their tiers differ.
            let bound = expected.max_buffer_size;
            assert_eq!(limits.max_buffer_size, bound);
            assert_eq!(
                limits.max_storage_buffer_binding_size,
                expected.max_storage_buffer_binding_size.min(bound) & !3
            );
            assert_eq!(
                limits.max_texture_dimension_2d,
                expected.max_texture_dimension_2d
            );
            assert_eq!(
                limits.max_storage_buffers_per_shader_stage,
                expected.max_storage_buffers_per_shader_stage
            );
            assert_eq!(
                compat.max_storage_buffers_in_fragment_stage,
                expected_compat.max_storage_buffers_in_fragment_stage
            );
        }
    }

    #[test]
    fn a_group_takes_the_highest_tier_all_of_its_limits_reach() {
        // An Apple GPU, roughly: generous everywhere, and between two tiers
        // in most groups.
        let (mut limits, mut compat) = tier(1);
        limits.max_texture_dimension_1d = 16384;
        limits.max_texture_dimension_2d = 16384;
        limits.max_texture_dimension_3d = 2048;
        limits.max_texture_array_layers = 2048;
        limits.max_storage_buffers_per_shader_stage = 31;
        compat.max_storage_buffers_in_fragment_stage = 31;
        compat.max_storage_buffers_in_vertex_stage = 31;
        limits.max_buffer_size = 3 * GIB + 12345;
        limits.max_storage_buffer_binding_size = 3 * GIB + 12345;
        limits.max_uniform_buffer_binding_size = 3 * GIB + 12345;
        limits.min_uniform_buffer_offset_alignment = 4;
        limits.min_storage_buffer_offset_alignment = 4;
        limits.max_compute_workgroup_storage_size = 40000;
        // One limit of its group falls short of the tier the others reach.
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_compute_workgroup_size_x = 1024;
        limits.max_compute_workgroup_size_y = 1000;
        limits.max_vertex_attributes = 31;
        limits.max_color_attachments = 8;
        apply(&mut limits, &mut compat);

        assert_eq!(limits.max_texture_dimension_2d, 16384);
        assert_eq!(limits.max_texture_array_layers, 2048);
        assert_eq!(limits.max_storage_buffers_per_shader_stage, 16);
        assert_eq!(compat.max_storage_buffers_in_fragment_stage, 16);
        assert_eq!(compat.max_storage_buffers_in_vertex_stage, 16);
        assert_eq!(limits.max_buffer_size, 2 * GIB);
        // 2 GiB - 4 is the tier, and the buffer size above bounds it anyway.
        assert_eq!(limits.max_storage_buffer_binding_size, 2 * GIB - 4);
        assert_eq!(limits.max_uniform_buffer_binding_size, 65536);
        // An alignment is better when smaller: it is rounded up.
        assert_eq!(limits.min_uniform_buffer_offset_alignment, 256);
        assert_eq!(limits.min_storage_buffer_offset_alignment, 256);
        assert_eq!(limits.max_compute_workgroup_storage_size, 32768);
        assert_eq!(limits.max_compute_invocations_per_workgroup, 256);
        assert_eq!(limits.max_compute_workgroup_size_x, 256);
        assert_eq!(limits.max_compute_workgroup_size_y, 256);
        assert_eq!(limits.max_vertex_attributes, 30);
        assert_eq!(limits.max_color_attachments, 8);

        // Tiering what is tiered changes nothing.
        let (again, again_compat) = {
            let (mut limits, mut compat) = (limits.clone(), compat.clone());
            apply(&mut limits, &mut compat);
            (limits, compat)
        };
        for limit in GROUPS.iter().copied().flatten() {
            assert_eq!(
                (limit.get)(&again, &again_compat),
                (limit.get)(&limits, &compat),
                "{}",
                limit.name
            );
        }
    }

    #[test]
    fn a_binding_size_is_held_to_the_buffer_size_it_is_tiered_beside() {
        // The binding size's tier is 4 GiB - 4 and the buffer size's 2 GiB.
        let (mut limits, mut compat) = tier(1);
        limits.max_buffer_size = 3 * GIB;
        limits.max_storage_buffer_binding_size = 4 * GIB - 1;
        apply(&mut limits, &mut compat);
        assert_eq!(limits.max_buffer_size, 2 * GIB);
        assert_eq!(limits.max_storage_buffer_binding_size, 2 * GIB);
    }

    #[test]
    fn a_request_within_the_tier_is_let_through_and_one_beyond_it_is_named() {
        let (allowed, allowed_compat) = tier(1);
        assert_eq!(exceeded(&undefined(), &allowed, &allowed_compat), None);
        assert_eq!(exceeded(&allowed, &allowed, &allowed_compat), None);

        let mut more = undefined();
        more.max_texture_dimension_2d = allowed.max_texture_dimension_2d + 1;
        let refusal = exceeded(&more, &allowed, &allowed_compat).expect("refused");
        assert!(refusal.contains("maxTextureDimension2D"), "{refusal}");
        assert!(
            refusal.contains("8193") && refusal.contains("8192"),
            "{refusal}"
        );

        // An alignment asks for more by being smaller.
        let mut finer = undefined();
        finer.min_uniform_buffer_offset_alignment = 64;
        let refusal = exceeded(&finer, &allowed, &allowed_compat).expect("refused");
        assert!(
            refusal.contains("minUniformBufferOffsetAlignment"),
            "{refusal}"
        );
        let mut coarser = undefined();
        coarser.min_uniform_buffer_offset_alignment = 512;
        assert_eq!(exceeded(&coarser, &allowed, &allowed_compat), None);

        // 4 GiB - 1 is a size, and what a 32-bit member means by "undefined".
        let mut large = undefined();
        large.max_buffer_size = u32::MAX as u64;
        assert!(exceeded(&large, &allowed, &allowed_compat).is_some());
    }

    #[test]
    fn compatibility_mode_limits_chained_to_a_request_are_checked_too() {
        let (allowed, allowed_compat) = tier(1);
        let mut required = undefined();
        required
            .chain
            .items
            .push(Chained::CompatibilityModeLimits(CompatibilityModeLimits {
                max_storage_buffers_in_vertex_stage: u32::MAX,
                max_storage_textures_in_vertex_stage: u32::MAX,
                max_storage_buffers_in_fragment_stage: allowed_compat
                    .max_storage_buffers_in_fragment_stage
                    + 1,
                max_storage_textures_in_fragment_stage: u32::MAX,
            }));
        let refusal = exceeded(&required, &allowed, &allowed_compat).expect("refused");
        assert!(
            refusal.contains("maxStorageBuffersInFragmentStage"),
            "{refusal}"
        );
    }
}
