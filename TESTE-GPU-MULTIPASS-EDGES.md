# Teste GPU: roughen_edges e spacer na rota de vários passes (AMD)

Estado: **nenhum dos dois efeitos roda na GPU ainda.** Ambos estão em `UNMEASURED_GPU_ROUTES`
(`crates/render/src/gpufx.rs`), e o classificador recusa os dois. Este teste existe para medir a
paridade antes de tirá-los da lista. Medir é pré-requisito, não formalidade.

Não há caso de paridade específico para estes efeitos no branch ainda. A ligação ao runner e o
harness genérico de paridade estão na thread da infra (`feat/gpu-multipass-route`). Quando ele
chegar, os comandos abaixo passam a valer. Até lá, a seção 2 é o único teste que roda.

## 1. Pré-requisitos

- Linux com GPU AMD (vendor 0x1002), driver Mesa/RADV ou amdgpu-pro.
- Rust estável recente e `cargo`.
- Repositório em `feat/gpu-multipass-edges`.

```bash
git clone https://github.com/Silvazrlq/filmcraft ~/filmcraft-edges
cd ~/filmcraft-edges
git checkout feat/gpu-multipass-edges
```

## 2. Gates sem GPU (devem passar antes de qualquer medição)

```bash
cargo fmt --all -- --check
cargo clippy -p filmcraft-render -p filmcraft-gpu --all-targets -- -D warnings
cargo xtask layers
cargo xtask assets
cargo test -p filmcraft-gpu -- noise_u64
cargo test -p filmcraft-render -- hash_emul gpufx_roughen gpufx_spacer
```

Critério: tudo verde. `noise_u64` valida os dois kernels com naga e não precisa de GPU.
`hash_emul` prova bit a bit que o hash em u32 bate com `hash_u` (4M entradas).

## 3. Paridade com GPU AMD (quando o harness genérico estiver no branch)

```bash
cargo test -p filmcraft-gpu fx_tests -- --nocapture
```

A saída precisa mostrar `Required FX parity adapter: <nome> vendor=0x1002`. Sem GPU AMD, o teste
falha de propósito. Qualquer outra GPU, ou nenhuma, é falha.

Casos a verificar, quando existirem no harness:

- `roughen_edges`: todos os `edge_type` (0..7), com borda pequena (≈ 0.5 px), média e grande;
  stretch negativo e positivo; `sharpness` e `influence` nos extremos.
- `spacer`: margens uniformes e assimétricas; escala acima de 1/2 (a única que vai para a GPU);
  raio de canto grande; `fill` ligado e desligado.

Critério de aceitação: diferença máxima por canal dentro da tolerância do harness. O blur do
roughen é a soma de caixas em ordem diferente da CPU (janela deslizante no CPU, soma direta na
GPU). A diferença esperada é de arredondamento em f32, não de algoritmo.

**Atenção:** a linha `skipping` não prova nada. Um teste que pula sem adaptador não mediu paridade.

## 4. Se a paridade passar

1. Retirar `roughen_edges` e `spacer` de `UNMEASURED_GPU_ROUTES` em `crates/render/src/gpufx.rs`.
2. Confirmar que o classificador passa a aceitá-los e que o teste de classificação continua verde.
3. Registrar o resultado (GPU, driver, saída do teste) neste arquivo.

## 5. Riscos conhecidos, sem medição

- **Custo do blur do roughen:** cada passe de caixa é O(r) por pixel; o CPU usa janela deslizante
  O(1). Com borda grande em 4K o passe pode estourar o limite de tempo do driver. Precisa de
  medição de tempo, não só de paridade.
- **Geometria do spacer duplicada:** `spacer_params` repete a aritmética de `vfx::distort::spacer`.
  Uma mudança futura no CPU não chega à GPU sozinha.
- **Coordenadas grandes no ruído:** o WGSL limita as células do ruído em ±2e9; o CPU usa i64. O
  stretch tem guarda de underflow, mas coordenadas muito grandes podem divergir.
