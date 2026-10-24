# Teste GPU: rota de vários passes (AMD)

Objetivo: medir na GPU AMD se o runner de vários passes (`crates/gpu/src/fx_multipass.rs`) produz a mesma imagem que a referência CPU. Nenhum efeito muda de rota neste branch: o teste mede só a infraestrutura.

## 1. Pré-requisitos

- Linux com GPU AMD (vendor 0x1002), driver Mesa/RADV ou amdgpu-pro.
- Rust estável recente e `cargo`.
- Repositório em `feat/gpu-multipass-route`.

```bash
git clone https://github.com/Silvazrlq/filmcraft ~/filmcraft-multipass
cd ~/filmcraft-multipass
git checkout feat/gpu-multipass-route
```

## 2. Gates sem GPU (devem passar antes do teste de GPU)

```bash
cargo fmt --all -- --check
cargo clippy -p filmcraft-render -p filmcraft-gpu --all-targets -- -D warnings
cargo xtask layers
cargo xtask assets
cargo test -p filmcraft-render gpufx_multipass
cargo test -p filmcraft-gpu fx_multipass
```

Critério: todos verdes. `kernel_contract_shader_validates_with_naga` valida o WGSL sem GPU.

## 3. Gate com GPU AMD

Os testes de paridade existentes exigem adaptador AMD e falham sem ele, de propósito:

```bash
cargo test -p filmcraft-gpu fx_tests -- --nocapture
```

Saída esperada: linha `Required FX parity adapter: <nome> vendor=0x1002`. Qualquer outra GPU ou nenhuma GPU é falha.

O teste `refuses_a_bad_plan_without_touching_the_device` roda com a GPU AMD quando disponível e pula sem adaptador:

```bash
cargo test -p filmcraft-gpu fx_multipass -- --nocapture
```

## 4. Critério de liberação

Liberar o próximo passo (portar efeitos) somente se:

1. Os gates da seção 2 passam.
2. Os testes `fx_multipass` passam na GPU AMD sem `skipping`.
3. Os testes de paridade de `fx_tests` que já passavam no 0.10 continuam passando sem alteração de tolerância.
4. Nenhuma tolerância foi relaxada para fazer passar.

Qualquer falha: parar, copiar o erro literal e o comando, e não contornar com fallback de CPU. O export é falha fechada.

## 5. Fora deste teste

- `vr_blur`, `vr_sharpen`, `vr_denoise`, `roughen_edges`, `spacer`: portados em threads separadas, a partir deste branch.
- `posterize_time`: depende de quadros de outros instantes. Fica sem rota GPU.
- `compound_blur`: depende de camada de trilha, `effect_requirements` não muda neste branch.
