# Teste GPU: roughen_edges e spacer na rota de vários passes (AMD)

Estado: **nenhum dos dois efeitos roda na GPU ainda.** Ambos estão em `UNMEASURED_GPU_ROUTES`
(`crates/render/src/gpufx.rs`), e o classificador recusa os dois. Este teste existe para medir a
paridade antes de tirá-los da lista. Medir é pré-requisito, não formalidade.

Os dois efeitos estão registrados no harness genérico (`MULTIPASS_ROUTES` em
`crates/render/src/gpufx_multipass.rs`; oráculo CPU em `crates/gpu/src/fx_tests.rs`, tolerância
`EXACT`). A seção 3 é a medição. Sem GPU AMD ela falha por desenho; isso não é paridade.

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

## 3. Paridade com GPU AMD

```bash
cargo test -p filmcraft-gpu multipass_routes_match_cpu_oracle_on_adapter -- --nocapture
```

Saída que importa: `roughen_edges: max rel diff ...` e `spacer: max rel diff ...`, com 0 valores
acima de `EXACT`. Sem adaptador o teste para com `FX parity gate requires a GPU adapter`.
Medição de referência, ainda não feita: nenhuma.

Parâmetros medidos: os de `multipass_case` (roughen com os padrões do projeto; spacer com margens
de 6 px e uniforme desligado, porque o padrão de 400 px cai no caminho de mip da CPU).

A saída precisa mostrar `Required FX parity adapter: <nome> vendor=0x1002`. Sem GPU AMD, o teste
falha de propósito. Qualquer outra GPU, ou nenhuma, é falha.

Cobertura atual: só um caso por efeito. Ainda não medido: `edge_type` 0..7, borda pequena e
grande, stretch negativo, margens assimétricas, `fill` ligado, raio grande. Esses casos precisam de
um harness com mais parâmetros; não estão cobertos.

Critério: diferença máxima por canal dentro de `EXACT`. Atenção: o blur do roughen soma caixas em
ordem diferente da CPU (janela deslizante no CPU, soma direta na GPU). Pode haver diferença de
arredondamento em f32 acima de `EXACT`. Se isso aparecer, a causa é a ordem da soma, não o
algoritmo, e a tolerância precisa de decisão explícita.

**Atenção:** a linha `skipping` não prova nada. Um teste que pula sem adaptador não mediu paridade.

## 4. Se a paridade passar

1. Retirar `roughen_edges` e `spacer` de `UNMEASURED_GPU_ROUTES` em `crates/render/src/gpufx.rs`.
   (O harness também exige que estejam na lista; `multipass_harness_refuses_ids_outside_unmeasured_list`
   falha se alguém tirar sem medir.)
2. Confirmar que o classificador passa a aceitá-los e que o teste de classificação continua verde.
3. Registrar o resultado (GPU, driver, saída do teste) neste arquivo.

## 5. Riscos conhecidos, sem medição

- **Tamanho do quadro fixo na montagem:** `multipass_effect_for(id, w, h)` recebe o tamanho; o
  integrador do runner precisa passar o tamanho real do quadro, não o do harness.
- **Custo do blur do roughen:** cada passe de caixa é O(r) por pixel; o CPU usa janela deslizante
  O(1). Com borda grande em 4K o passe pode estourar o limite de tempo do driver. Precisa de
  medição de tempo, não só de paridade.
- **Geometria do spacer duplicada:** `spacer_params` repete a aritmética de `vfx::distort::spacer`.
  Uma mudança futura no CPU não chega à GPU sozinha.
- **Coordenadas grandes no ruído:** o WGSL limita as células do ruído em ±2e9; o CPU usa i64. O
  stretch tem guarda de underflow, mas coordenadas muito grandes podem divergir.
