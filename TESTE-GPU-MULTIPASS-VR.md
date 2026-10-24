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
cargo test -p filmcraft-gpu --lib -- vr_
```

- `row_tests::vr_plans_match_the_cpu_reference_for_every_layout`: executa cada plano com a semântica dos kernels na CPU e compara com a referência CPU por vista (layouts 0, 1, 2; tamanhos 16×8 e 9×7; blur, sharpen, denoise, denoise com show). Isso prova a montagem do plano e a matemática, não o shader.
- `vr_wgsl_validates`: naga valida `vr.wgsl`. `vr_wgsl_exposes_every_plan_kernel`: os nomes batem com o plano.

## Pendente antes de medir na AMD

1. Caso de paridade GPU para cada id, pelo harness genérico que a thread da infra (`feat/gpu-multipass-route`) publica. Depois de ligar os ids a esse caminho, eles saem de `UNMEASURED_GPU_ROUTES` só após medição.
2. Medição na AMD:

```
CRAFT_FONTS_DIR="$PWD/../craft-fonts" cargo test -p filmcraft-gpu --lib -- --nocapture vr_
```

`fx_tests::gpu_vr_effects_report_pixel_deltas_against_independent_and_typed_cpu` (da infra) exige adaptador AMD. Sem AMD ele falha; isso não é paridade. Não conta como passou.
