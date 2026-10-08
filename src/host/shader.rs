//! `WGPUShaderModule`.

use std::borrow::Cow;
use std::sync::Arc;

use super::buffer::label;
use super::{DeviceState, Host, Object, ObjectKind, Record};
use crate::abi::api::ShaderModuleApi;
use crate::abi::enums::{CompilationInfoRequestStatus, CompilationMessageType};
use crate::abi::structs::ShaderModuleDescriptor;
use crate::abi::{AbiEnum, Cx, EVENT_COMPILATION_INFO, Handle};

pub(crate) struct ShaderState {
    pub(crate) module: wgpu::ShaderModule,
    #[allow(dead_code)]
    pub(crate) device: Arc<DeviceState>,
}

impl Host {
    pub(crate) fn create_shader_module(
        &self,
        device: Handle,
        descriptor: &ShaderModuleDescriptor,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let source = if let Some(wgsl) = descriptor.chain.shader_source_wgsl() {
            wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl.code.as_str()))
        } else {
            // WGSL is the one shading language every host of this ABI takes
            // (a browser accepts nothing else), so it is the only one any of
            // them takes: a program that runs here runs there.
            self.validation(
                &device_state,
                if descriptor.chain.shader_source_spirv().is_some() {
                    "SPIR-V shader sources are not supported \
                     (WGPUInstanceFeatureName_ShaderSourceSPIRV is not available)."
                } else {
                    "A shader module descriptor must chain WGPUShaderSourceWGSL."
                },
            );
            return 0;
        };
        let module = device_state
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: label(&descriptor.label),
                source,
            });
        let state = ShaderState {
            module,
            device: device_state,
        };
        self.insert(
            Object::ShaderModule(Arc::new(state)),
            self.instance_of(device),
        )
    }

    /// Starts `wgpuShaderModuleGetCompilationInfo`. The messages travel in the
    /// event record's extra payload:
    ///
    /// ```text
    /// u32 count; u32 reserved;
    /// per message:
    ///   u32 type; u32 messageLength;
    ///   u64 lineNum; u64 linePos; u64 offset; u64 length;
    ///   u8  message[messageLength], padded to 8 bytes
    /// ```
    pub(crate) fn shader_module_get_compilation_info(&self, module: Handle, future: u64) {
        let mut extra = Vec::new();
        let mut count = 0u32;
        if let Some(state) = self.shader_module(module) {
            let info = pollster::block_on(state.module.get_compilation_info());
            for message in &info.messages {
                let ty = match message.message_type {
                    wgpu::CompilationMessageType::Error => CompilationMessageType::Error,
                    wgpu::CompilationMessageType::Warning => CompilationMessageType::Warning,
                    wgpu::CompilationMessageType::Info => CompilationMessageType::Info,
                };
                let (line, position, offset, length) =
                    message.location.map_or((0, 0, 0, 0), |at| {
                        (
                            u64::from(at.line_number),
                            u64::from(at.line_position),
                            u64::from(at.offset),
                            u64::from(at.length),
                        )
                    });
                extra.extend_from_slice(&ty.as_u32().to_le_bytes());
                extra.extend_from_slice(&(message.message.len() as u32).to_le_bytes());
                for value in [line, position, offset, length] {
                    extra.extend_from_slice(&value.to_le_bytes());
                }
                extra.extend_from_slice(message.message.as_bytes());
                extra.resize((extra.len() + 7) & !7, 0);
                count += 1;
            }
        }
        let mut payload = Vec::with_capacity(extra.len() + 8);
        payload.extend_from_slice(&count.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&extra);
        self.events.push(Record {
            future,
            kind: EVENT_COMPILATION_INFO,
            status: CompilationInfoRequestStatus::Success.as_u32(),
            extra: payload,
            ..Record::default()
        });
    }
}

impl ShaderModuleApi for Host {
    fn shader_module_set_label(&self, _cx: &mut Cx<'_>, shader_module: Handle, _label: &str) {
        let _ = self.shader_module(shader_module);
    }

    fn shader_module_add_ref(&self, _cx: &mut Cx<'_>, shader_module: Handle) {
        self.add_ref(shader_module, ObjectKind::ShaderModule);
    }

    fn shader_module_release(&self, _cx: &mut Cx<'_>, shader_module: Handle) {
        self.release(shader_module, ObjectKind::ShaderModule);
    }
}
