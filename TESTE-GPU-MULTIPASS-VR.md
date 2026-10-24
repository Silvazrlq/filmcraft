# Teste GPU de vários passes: vr_blur, vr_sharpen, vr_denoise

Status: **não medido**. Os três ids estão em `UNMEASURED_GPU_ROUTES` (`crates/render/src/gpufx.rs`) e o classificador (`evaluate_gpu_effect`) os recusa. Continuam na CPU. Nenhum caso de paridade GPU existe ainda, então nada aqui prova paridade.

## O que já é verificado sem GPU

```
cargo test -p filmcraft-gpu --lib -- vr_wgsl_validates
cargo test -p filmcraft-render --lib -- row_tests vr_
cargo test -p filmcraft-render --lib -- unmeasured_routes_are_never_in_the_gpu_allowlist
```

- `vr_wgsl_validates`: `crates/gpu/src/vr.wgsl` faz parse e validação com naga. Não prova que o kernel calcula o mesmo que a CPU.
- `row_radius`, `blur_row_wrap` e `blur_rows_wrap` (render, `vfx::immersive::row_tests`): testes próprios, não cobertos pelo filtro `vr_`.
- `vr_*` (render): oráculos CPU de VR (chromatic, color gradients, projeção, esfera) e rotação. Não exercitam o caminho de blur refatorado.
- `unmeasured_routes_are_never_in_the_gpu_allowlist`: os três ids não são elegíveis.

## Pendente antes de medir na AMD

1. Contrato de bindings: `crates/gpu/src/fx_multipass.wgsl` tem só as entradas 0–2. Os kernels VR precisam de um uniform com raio, retângulo da vista, `sigma`, `amount`, `threshold` e `show`. Sem isso, nenhum passo roda.
2. Caso de paridade em `crates/gpu/src/fx_tests.rs` para cada id. Só depois disso dá para medir.
3. Medição na AMD (VA-API/Vulkan), com o caso de paridade rodando de fato:

```
CRAFT_FONTS_DIR="$PWD/../craft-fonts" cargo test -p filmcraft-gpu --lib -- --nocapture vr_
```

Critério: o caso não pode aparecer como `skipping`. Se o device não existir, o teste falha ou não conta.
