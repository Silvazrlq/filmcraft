# Teste GPU de vários passes: vr_blur, vr_sharpen, vr_denoise

Status: **não medido**. Os três ids estão em `UNMEASURED_GPU_ROUTES` (`crates/render/src/gpufx.rs`) e `evaluate_gpu_effect` os recusa. Seguem na CPU. Nada aqui prova paridade GPU.

## O que existe

- `vfx::immersive::vr_plan(VrKind, layout, w, h)`: plano `MultipassPlan` com parâmetros por passe (`Pass.params`). Kernels: `vr_blur_row` (3 passes de caixa por linha, com wrap), `vr_box_col` (passes verticais com repeat), `vr_copy`, `vr_sharpen_combine`, `vr_denoise_combine`.
- `crates/gpu/src/vr.wgsl`: os kernels acima, no contrato de bindings (0, 1 entradas; 2 saída; 3 uniform de parâmetros).
- `row_radius`, `box_row_wrap`, `blur_row_wrap` são a especificação CPU dos kernels.

## Verificado sem GPU

```
cargo test -p filmcraft-render --lib -- row_tests
cargo test -p filmcraft-render --lib -- unmeasured_routes_are_never_in_the_gpu_allowlist
cargo test -p filmcraft-gpu --lib -- multipass_routes_match_cpu_oracle_on_adapter --nocapture
```

- `row_tests::vr_plans_match_the_cpu_reference_for_every_layout`: executa cada plano com a semântica dos kernels na CPU e compara com a referência CPU por vista (layouts 0, 1, 2; tamanhos 16×8 e 9×7; blur, sharpen, denoise, denoise com show). Isso prova a montagem do plano e a matemática, não o shader.
- `vr_wgsl_validates`: naga valida `vr.wgsl`. `vr_wgsl_exposes_every_plan_kernel`: os nomes batem com o plano.

## Harness genérico (registrado)

- `MULTIPASS_ROUTES` (`crates/render/src/gpufx_multipass.rs`): `vr_blur`, `vr_sharpen`, `vr_denoise`.
- `multipass_effect_for(id, w, h)`: monta o plano no tamanho do frame, com parâmetros de amostra fixos (blur sigma 2.0; sharpen 0.6 / sigma 1.2; denoise sigma 0.9 / threshold 0.08). Os parâmetros ainda não vêm da instância do efeito.
- Oráculo CPU (`cpu_reference` em `fx_tests.rs`): `vfx::apply` com os mesmos parâmetros de amostra, tolerância `EXACT`.
- Os ids seguem em `UNMEASURED_GPU_ROUTES`. `measure_route` recusa qualquer id fora dessa lista.

## Pendente antes de medir na AMD

1. Medição na AMD:

```
CRAFT_FONTS_DIR="$PWD/../craft-fonts" cargo test -p filmcraft-gpu --lib -- --nocapture vr_
```

`fx_tests::multipass_routes_match_cpu_oracle_on_adapter` e `fx_tests::gpu_vr_effects_report_pixel_deltas_against_independent_and_typed_cpu` exigem adaptador AMD. Sem AMD falham com `NotFound` (verificado neste container). Isso não é paridade e não conta como passou.
2. Ligar os parâmetros reais da instância do efeito, e só então o caminho do efeito; até lá a medição vale só para os parâmetros de amostra.
